#ifndef H_COMMON_DEBUG
#define H_COMMON_DEBUG

#ifdef FLAG_DEBUG
#undef NDEBUG
#define DEBUG 1
#endif

#if defined(FLAG_DEBUG) && defined(FLAG_PRINTD)
/*#define printd(fmt, ...) printf(fmt, __VA_ARGS__);*/
#define printd(args) printf args
#else
#define printd(args) /**/
#endif

#if defined(FLAG_DEBUG)
#define exitd(flag) exit(flag);
#else
#define exitd(flag) /**/
#endif

#if defined(FLAG_DEBUG) && defined(FLAG_ASSERTD)
#define assertd(condition) \
    if (!(condition)) { \
        printd(("\n")); \
        printd(("ERROR: on line %d, in file %s:\n", __LINE__, __FILE__ ));\
        printd(("       assertion triggered, the following condition does not hold: " #condition "\n")); \
        assert(condition); \
        exitd(1); \
    }
#else
#define assertd(condition) /**/
#endif

#define assertd_is_square(mat) assertd(is_square(mat));
#define assertd_same_dims(mat1, mat2) assertd((mat1->h == mat2->h) && (mat1->w == mat2->w));

#endif
