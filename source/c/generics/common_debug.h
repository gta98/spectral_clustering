#ifndef H_COMMON_DEBUG
#define H_COMMON_DEBUG

#if defined(FLAG_DEBUG) && defined(FLAG_PRINTD)
/*#define printd(fmt, ...) printf(fmt, __VA_ARGS__);*/
#define printd printf
#else
#define printd(fmt, ...) /**/
#endif

#if defined(FLAG_DEBUG) && defined(FLAG_ASSERTD)
#define assertd(condition) assert(condition);
#else
#define assertd(condition) /**/
#endif

#define assertd_is_square(mat) assertd(is_square(mat));
#define assertd_same_dims(mat1, mat2) assertd((mat1->h == mat2->h) && (mat1->w == mat2->w));

#endif