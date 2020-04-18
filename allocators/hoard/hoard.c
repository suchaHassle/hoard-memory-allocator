#include "memlib.h"
#include "mm_thread.h"
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

#define NUM_BINS 6
#define SZ_CLASS 9

#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

#define GET_SZ_CLASS(x) (((x) > 8) ? log2floor(x) - 2 : 0)
#define LOCK(x) (pthread_spin_lock(&((x)->lock)))
#define UNLOCK(x) (pthread_spin_unlock(&((x)->lock)))
#define TRYLOCK(x) (pthread_spin_trylock(&((x)->lock)))
#define PAGE_ALIGN(x)                                                          \
  (((unsigned long long)(x) >> LOG_PAGE_SIZE) << LOG_PAGE_SIZE)

typedef struct superblock {
  pthread_spinlock_t lock;
  u_int32_t in_use;
  u_int8_t bin_idx;         // Index in heap.bins[self.sz_idx]
  u_int8_t sz_idx;          // Size class, real size is 2^(sz_idx + 3)
  u_int8_t heap_owner;      // The owning heap.heap_idx
  struct superblock *next;
  struct superblock *prev;
  u_int8_t num_pages;       // Number of pages, only for huge pages

  // Place bitmap at end of struct and aligned to a cacheline, so that when
  // the [lock] field is fetched, the prefetcher makes the bitmap be fetched
  // by the time it's needed.
  u_int8_t __attribute__((aligned(64)) bitmap[64];
} __attribute__((aligned(64))) superblock_t;

typedef struct heap {
  u_int8_t heap_idx;
  int in_use;          // Bytes used; u_i in Hoard
  int pages_allocated; // Bytes allocates in pages; a_i in Hoard
  pthread_spinlock_t lock;
  superblock_t *bins[SZ_CLASS][NUM_BINS];  // Superblocks by size and fullness
} __attribute__((aligned(64))) heap_t;

static int NUM_PROCS;
static u_int64_t PAGE_SIZE;
// Maintain the logarithm of the page size so that most operations can use a
// shift rather than a division operation.
static u_int64_t LOG_PAGE_SIZE;
static int K = 8;
static float F = 0.25;
static pthread_spinlock_t new_page_lock;
static heap_t *heaps;
static superblock_t *totally_free_superblocks = NULL;

// It seems that on some machines [getTID] is very slow and accounts for 30%
// of program time. This is not the case on others, like wolf or yelp.
// Instead of relying on machine configuration to be in our favour, we cache
// the computed hash in the TLS block.
__thread int tls_hash = 0;
static inline int hash() {
  return tls_hash ? tls_hash : (tls_hash = (getTID() % NUM_PROCS) + 1);
}

static inline int log2floor(u_int64_t sz) {
  return sizeof(u_int64_t) * 8 - 1 - __builtin_clzll(sz);
}

#define to_log_size(idx) (3 + (idx))
static inline u_int64_t to_size(int idx) { return 1ULL << to_log_size(idx); }

static inline int num_blocks(int sz_idx) {
  // This shift is just division by [to_size(sz_idx)], but GCC isn't smart
  // enough to realize that [to_size] is always a power of two and turn it
  // into a shift itself.
  return (PAGE_SIZE - sizeof(superblock_t)) >> to_log_size(sz_idx);
}

static inline bool is_hugeblock(superblock_t *sb) { return sb->num_pages > 0; }

static inline u_int64_t bitmask_idx(void *ptr, superblock_t *sb) {
  u_int64_t offset = ((unsigned long long)ptr) - (unsigned long long)sb;
  assert(offset >= sizeof(superblock_t));

  int ret = (offset - sizeof(superblock_t)) >> to_log_size(sb->sz_idx);
  assert(ret >= 0);
  return ret;
}

static inline int next_block(superblock_t *sb, int blocks_per_sb) {
  // Access bitmap as 64-bit words for fewer loop jumps. This is safe to do
  // because the bitmap is aligned on a 64-byte boundary.
  u_int64_t *lb = (u_int64_t *)sb->bitmap;

  for (int i = 0; i < 64 / sizeof(u_int64_t); i++) {
    int j = __builtin_ffsll(~lb[i]);
    if (j == 0)
      continue;

    int block = i * 8 * sizeof(u_int64_t) + j - 1;
    if (block >= blocks_per_sb)
      return -1;
    return block;
  }

  return -1;
}

static inline bool is_superblock_full(superblock_t *sb, int sz_class_idx) {
  int ret =
      sb->in_use + to_size(sz_class_idx) > (PAGE_SIZE - sizeof(superblock_t));
  assert((next_block(sb, num_blocks(sz_class_idx)) == -1) == ret);
  return ret;
}

static inline void unlink_superblock(superblock_t **head, superblock_t *sb) {
  if (*head == sb)
    *head = sb->next;
  if (sb->prev)
    sb->prev->next = sb->next;
  if (sb->next)
    sb->next->prev = sb->prev;
}

static inline void *create_new_hugeblock(size_t sz) {
  int num_pages = (sz / (PAGE_SIZE - sizeof(superblock_t))) + 1;
  pthread_spin_lock(&new_page_lock);
  superblock_t *sb = (superblock_t *)mem_sbrk(PAGE_SIZE * num_pages);
  pthread_spin_unlock(&new_page_lock);
  sb->num_pages = num_pages;

  return (char *)sb + sizeof(superblock_t);
}

static inline void free_hugeblock(superblock_t *sb) {
  int num_pages = sb->num_pages;

  pthread_spin_lock(&new_page_lock);
  for (int i = 0; i < num_pages; i++) {
    superblock_t *new_sb = (((char *)sb) + (PAGE_SIZE * i));
    new_sb->next = totally_free_superblocks;
    new_sb->prev = NULL;

    if (new_sb->next)
      new_sb->next->prev = new_sb;
    totally_free_superblocks = new_sb;
  }
  pthread_spin_unlock(&new_page_lock);
}

static void move_superblock(heap_t *old, heap_t *new, superblock_t *sb,
                            int sz_class_idx, int bin) {
  assert(sb->in_use <= PAGE_SIZE);
  assert(bin >= 0);

  // We use a special bin to denote totally full blocks.
  int new_bin = is_superblock_full(sb, sz_class_idx)
                    ? NUM_BINS - 1
                    : ((NUM_BINS - 1) * sb->in_use) >> LOG_PAGE_SIZE;
  if (bin != new_bin || new != NULL) {
    // Unlink the superblock from the old heap.
    unlink_superblock(&old->bins[sz_class_idx][bin], sb);

    sb->bin_idx = new_bin;
    sb->prev = NULL;

    // Link the superblock into the new heap, if any.
    heap_t *heap = (new != NULL) ? new : old;
    sb->next = heap->bins[sz_class_idx][new_bin];
    if (sb->next)
      sb->next->prev = sb;
    heap->bins[sz_class_idx][new_bin] = sb;
  }

  // It's possible we were called with [old == new] to move the superblock to
  // the front of its fullness group, in which case we don't have to update any
  // statistics.
  if (old != new &&new != NULL) {
    old->in_use -= to_size(sb->sz_idx);
    new->in_use += to_size(sb->sz_idx);
    old->pages_allocated--;
    new->pages_allocated++;
  }
}

superblock_t *create_new_superblock(heap_t *heap, int sz_class_idx) {
  superblock_t *sb;
  pthread_spin_lock(&new_page_lock);
  if (totally_free_superblocks != NULL) {
    sb = totally_free_superblocks;
    totally_free_superblocks = sb->next;
  } else {
    sb = mem_sbrk(PAGE_SIZE); // Non-atomic op
  }
  pthread_spin_unlock(&new_page_lock);

  if (sb == NULL) {
    fprintf(stderr, "failed to allocate space for superblock\n");
    abort();
  }

  memset(sb, 0, sizeof(superblock_t));

  // Link the superblock into the heap.
  sb->heap_owner = heap->heap_idx;
  sb->sz_idx = sz_class_idx;
  sb->prev = NULL;
  sb->next = heap->bins[sz_class_idx][0];
  if (sb->next)
    sb->next->prev = sb;
  heap->bins[sz_class_idx][0] = sb;
  heap->pages_allocated++;

  pthread_spin_init(&sb->lock, PTHREAD_PROCESS_PRIVATE);
  LOCK(sb);

  return sb;
}

superblock_t *get_superblock_from_global(heap_t *heap, int sz_class_idx) {
  // Start from [NUM_BINS - 2] to avoid the totally full bin.
  for (int i = NUM_BINS - 2; i >= 0; i--) {
    for (superblock_t *sb = heaps->bins[sz_class_idx][i]; sb; sb = sb->next) {
      if (likely(TRYLOCK(sb) == 0)) {
        // Is it still not full now?
        if (!is_superblock_full(sb, sz_class_idx)) {
          move_superblock(heaps, heap, sb, sz_class_idx, i);
          sb->heap_owner = heap->heap_idx;
          sb->bin_idx = i;
          return sb;
        }

        UNLOCK(sb);
      }
    }
  }
  return NULL;
}

superblock_t *get_superblock_from_heap(heap_t *heap, int sz_class_idx) {
  for (int i = NUM_BINS - 2; i >= 0; i--) {
    for (superblock_t *sb = heap->bins[sz_class_idx][i]; sb; sb = sb->next) {
      if (likely(TRYLOCK(sb) == 0)) {
        if (!is_superblock_full(sb, sz_class_idx)) {
          sb->bin_idx = i;
          return sb;
        }

        UNLOCK(sb);
      }
    }
  }

  return NULL;
}

superblock_t *get_superblock_and_lock(heap_t *heap, int sz_class_idx) {
  superblock_t *sb = get_superblock_from_heap(heap, sz_class_idx);
  if (sb)
    return sb;

  LOCK(heaps);
  sb = get_superblock_from_global(heap, sz_class_idx);
  UNLOCK(heaps);
  if (sb)
    return sb;

  return create_new_superblock(heap, sz_class_idx);
}

void *mm_malloc(size_t sz) {
  if (sz > PAGE_SIZE / 2) {
    void *ptr = create_new_hugeblock(sz);
    return ptr;
  }

  int heap_id = hash();
  heap_t *heap = &heaps[heap_id];
  assert(heap->heap_idx == heap_id);

  int sz_class_idx = GET_SZ_CLASS(sz);
  assert(sz_class_idx < SZ_CLASS);

  LOCK(heap);
  superblock_t *sb = get_superblock_and_lock(heap, sz_class_idx);

  int idx = next_block(sb, num_blocks(sz_class_idx));
  assert(idx >= 0);

  sb->bitmap[idx / 8] |= (1 << (idx % 8));
  sb->in_use += to_size(sz_class_idx);
  heap->in_use += to_size(sz_class_idx);

  move_superblock(heap, NULL, sb, sz_class_idx, sb->bin_idx);

  UNLOCK(sb);
  UNLOCK(heap);

  return ((char *)sb) + sizeof(superblock_t) + (idx * to_size(sz_class_idx));
}

void mm_free(void *ptr) {
  superblock_t *sb = (superblock_t *)PAGE_ALIGN(ptr);
  if (is_hugeblock(sb)) {
    free_hugeblock(sb);
    return;
  }

  int heap_owner;
// For lock ordering purposes, we must always grab a heap lock before a
// superblock lock. However, this means that between when the heap is locked
// and when the superblock is locked, the superblock could have been moved
// by another thread into another heap, so any operation we were about to
// perform is invalid.
retry_lock:
  heap_owner = sb->heap_owner;
  heap_t *heap = &heaps[heap_owner];
  LOCK(heap);
  LOCK(sb);
  if (unlikely(sb->heap_owner != heap_owner)) {
    // This race happens very infrequently even with 12 cores, but not bailing
    // out here would deadlock. Instead, pay the performance penalty, unlock
    // everything and try again.
    UNLOCK(sb);
    UNLOCK(heap);
    goto retry_lock;
  }

  u_int64_t location = bitmask_idx(ptr, sb);
  sb->bitmap[location / 8] &= ~(1 << (location % 8));
  sb->in_use -= to_size(sb->sz_idx);
  heap->in_use -= to_size(sb->sz_idx);

  // Move the superblock to its appropriate fullness group.
  move_superblock(heap, heap, sb, sb->sz_idx, sb->bin_idx);

  // Unlock the superblock here, even though we may end up immediately
  // reacquiring it below, so we don't have lock inversion with the global
  // heap lock.
  UNLOCK(sb);

  // If the superblock is not in the global heap already, and meets the
  // emptiness threshold, transfer a mostly-empty superblock from this
  // heap into the global heap.
  if (heap_owner != 0 && heap->in_use < heap->pages_allocated - K &&
      heap->in_use < (1 - F) * heap->pages_allocated * PAGE_SIZE) {
    assert(heaps != heap);
    LOCK(heaps);
    // Try moving a superblock. We'll first try to move the first (least
    // full) entry, but try subsequent ones if we contend on that
    // superblock's lock.
    for (int i = 0; i < SZ_CLASS; i++) {
      superblock_t *s1 = heap->bins[i][0];
      if (s1 == NULL || TRYLOCK(s1) != 0)
        continue;

      if (s1->in_use == 0) {
        unlink_superblock(&heap->bins[i][0], s1);
        heap->pages_allocated--;
        UNLOCK(s1);
        UNLOCK(heaps);
        UNLOCK(heap);

        pthread_spin_destroy(&s1->lock);
        pthread_spin_lock(&new_page_lock);
        s1->next = totally_free_superblocks;
        totally_free_superblocks = s1;
        pthread_spin_unlock(&new_page_lock);
        return;
      } else {
        // Transfer the superblock from a thread heap into the global heap.
        move_superblock(heap, heaps, s1, i, 0);
        s1->heap_owner = 0;
        UNLOCK(s1);
        break;
      }
    }

    UNLOCK(heaps);
  }

  UNLOCK(heap);
}

int mm_init(void) {
  if (mem_init() == -1) {
    fprintf(stderr, "Failed to initialize memory\n");
    return -1;
  }

  pthread_spin_init(&new_page_lock, PTHREAD_PROCESS_PRIVATE);

  NUM_PROCS = getNumProcessors();
  PAGE_SIZE = mem_pagesize();
  LOG_PAGE_SIZE = log2floor(PAGE_SIZE);
  int num_pages = (NUM_PROCS / (PAGE_SIZE / sizeof(heap_t))) + 1;

  heaps = (heap_t *)mem_sbrk(PAGE_SIZE * num_pages);
  assert(heaps != NULL);

  for (int i = 0; i <= NUM_PROCS; i++) {
    heap_t *h = &heaps[i];
    pthread_spin_init(&h->lock, PTHREAD_PROCESS_PRIVATE);
    h->heap_idx = i;
    h->in_use = 0;
    h->pages_allocated = 0;
    for (int x = 0; x < SZ_CLASS; x++) {
      for (int y = 0; y < NUM_BINS; y++) {
        h->bins[x][y] = NULL;
      }
    }
  }

  return 0;
}
