#include "matrix_reader.h"

/*
locates needle in file
*fh: file handler
needle: character to look for
*location: where to store location if found, or chars to eof if not
return value: true iff found needle
*/
bool FILE_locate_needle_or_EOF(FILE* fh, char needle, int* location) {
    char c;
    long start;
    *location = 0;
    start = ftell(fh);
    while (1) {
        c = fgetc(fh);
        if ((c == EOF) || (c == needle)) {
            break;
        }
        (*location)++;
    }
    fseek(fh, start, SEEK_SET);

    return (c == needle);
}

/*
advance fh by min(length, dist to EOF) chars
*fh: file handler
length: maximum number of characters to advance
return value: actual number of characters advanced
*/
int FILE_advance(FILE* fh, int length) {
    int counted = 0;
    while (counted < length) {
        if (fgetc(fh) == EOF) break;
        counted++;
    }
    return counted;
}

bool is_char_meaningful(char c) {
    return ((c != '\n') && (c != ' ') && (c != '\t') && (c != EOF));
}

bool is_char_meaningless(char c) {
    return !is_char_meaningful(c);
}

/*
skips meaningless characters - {'\n', '\t', ' '}
return value: true iff there is anything left to read
*/
bool FILE_skip_meaningless(FILE* fh) {
    char c;
    do {
        c = fgetc(fh);
        /* detected a meaningful character? */
        if (is_char_meaningful(c)) {
            fseek(fh, -1L, SEEK_CUR); /* so we can still read c */
            return true;
        }
    } while (c != EOF);

    /* if we are here, it means that we have reached EOF */
    return false;
}

/*
skips meaningful characters - anything that is not meaningless
return value: true iff there is anything left to read
*/
bool FILE_skip_meaningful(FILE* fh) {
    char c;
    do {
        c = fgetc(fh);
        if (c == EOF) break;
        /* detected a meaningless character? */
        if (is_char_meaningless(c)) {
            fseek(fh, -1L, SEEK_CUR); /* so we can still read c */
            return true;
        }
    } while (c != EOF);

    /* if we are here, it means that we have reached EOF */
    return false;
}



/*
reads up to length chars from fh and puts them in dst
**dst: pointer to save location
*fh: file handler
length: maximum number of characters to read
return value: actual number of characters saved to *dst, or -1 if could not allocate
*/
int FILE_get(char** dst, FILE* fh, int length) {
    char* line;
    long start;
    int counted, total_read;
    char c;

    start = ftell(fh);
    counted = 0;
    while (counted < length) {
        c = fgetc(fh);
        if (c == EOF) break;
        counted++;
    }
    fseek(fh, start, SEEK_SET);

    line = malloc(/*sizeof(char) * */(counted + 1));
    if (line == NULL) return -1;

    total_read = 0;
    while (total_read < counted) {
        c = fgetc(fh);
        line[total_read] = c;
        total_read++;
    }
    line[counted] = 0;
    *dst = line;

    return counted;
}

bool would_negative_double_overflow(double x) {
    double y;
    y = ((double)(-1)) * x;
    y = ((double)(-1)) * y;
    if (y != x) return true;
    else return false;
}

bool c_in_range(char c, int i, int j) {
    assertd(i <= j);
    assertd((0 <= i) && (j <= 9));
    return (i <= (c-'0')) && ((c-'0') <= j);
}

typedef enum {
    SUCCESS_VALID_CONTENT,
    NOTHING_LEFT_TO_READ,
    INVALID_NUMBER_TWO_DOTS,
    INVALID_NUMBER_BEFORE_AFTER_DOT,
    INVALID_NUMBER_NONNUMERIC,
    INVALID_NUMBER_FSCANF,
    INVALID_NUMBER_NEG_OVERFLOW
} file_read_enum;

/*
return value: true iff success 
*/
file_read_enum FILE_get_next_num(FILE* fh, double* num) {
    bool anything_left_to_read;
    char c;
    long loc_start, loc_finish;
    long loc_first_nonzero, loc_dot/*, loc_last_nonzero*/;
    bool is_signed;
    bool is_negative;
    bool had_e;
    int count_successfully_fscanned;

    anything_left_to_read = FILE_skip_meaningless(fh);
    if (!anything_left_to_read) {
        return NOTHING_LEFT_TO_READ;
    }

    loc_start = ftell(fh);
    is_signed = false;
    is_negative = false;
    had_e = false;
    loc_dot = -1;
    loc_first_nonzero = -1;
    do {
        c = fgetc(fh);

        if (c_in_range(c, 1,9)) {
            if (loc_first_nonzero < 0) loc_first_nonzero = ftell(fh);
            /*loc_last_nonzero = ftell(fh);*/
        }
        
        if (c_in_range(c, 0, 9)) {
            /* nothing interesting here - we are only measuring positions */
        }
        else if ((c == '-') && (ftell(fh) == (loc_start+1))) {
            is_signed = true;
            is_negative = true;
        }
        else if ((c == '+') && (ftell(fh) == (loc_start+1))) {
            is_signed = true;
        }
        else if (c == '.') {
            if (loc_dot >= 0) return INVALID_NUMBER_TWO_DOTS;
            loc_dot = ftell(fh);
        }
        else if ((c == ',') || (c == '\n') || (c == EOF)) {
            /* finished reading entire number */
            break;
        } else if ((c == 'e') || (c == 'E')) {
            had_e = true;
        } else if (((c == '-') || (c == '+')) && had_e) {
            /* may god bE with mE */
            /* prevent throwing invalid */
            /* this is not meant to be completely failsafe */
        }
        else {
            return INVALID_NUMBER_NONNUMERIC;
        }
    } while (1);
    loc_finish = ftell(fh);

    fseek(fh, loc_start, SEEK_SET);

    if (is_signed) {
        /* we will read, and then (maybe) flip */
        fseek(fh, 1, SEEK_CUR);
    }

    count_successfully_fscanned = fscanf(fh, "%lf", num);
    if (count_successfully_fscanned != 1) {
        return INVALID_NUMBER_FSCANF;
    }

    if (is_negative) {
        if (would_negative_double_overflow(*num)) {
            return INVALID_NUMBER_NEG_OVERFLOW;
        } else {
            *num = (*num) * ((double)(-1));
        }
    }

    fseek(fh, loc_finish, SEEK_SET);
    return SUCCESS;
}

int get_number_of_dimensions_in_line(FILE* fh) {
    int i, d, chars_in_first_line;
    long start;

    start = ftell(fh);

    /* find number of chars in current line */
    FILE_locate_needle_or_EOF(fh, '\n', &chars_in_first_line);
    /* if nothing found, we have 0 characters and thus 0 dims */
    if (chars_in_first_line == 0) return 0;

    /* otherwise, we have more than one char in line */

    /* so count 1+commas */
    for (i=0, d=1; i<chars_in_first_line; i++) {
        if (fgetc(fh) == ',') d++;
    }
    /* assuming all numbers are valid, this should be the number of dims in line */
    /* however, this does not guarantee validity in any way, shape or form */

    /* reset - go back to line start */
    fseek(fh, start, SEEK_SET);

    /* number of dimensions, assuming all lines have the same dimension */
    return d;
}

/* returns true iff all lines have the same number of commas */
bool is_number_of_dimensions_consistent(FILE* fh, int d) {
    bool anything_left_to_read;
    bool did_find_error;
    int d_cur;

    did_find_error = false;

    do {
        /* skip all newline characters */
        anything_left_to_read = FILE_skip_meaningless(fh);
        /* if nothing left to read, we are done */
        if (!anything_left_to_read) break;

        /* get dimensions in line pointed to by fh */
        d_cur = get_number_of_dimensions_in_line(fh);

        /* we wanna make sure d_cur is consistent */
        if (d_cur != d) {
            did_find_error = true;
            break;
        }

        /* if consistent - just skip whatever is in the current line */
        anything_left_to_read = FILE_skip_meaningful(fh);
        /* if nothing left to read, we are done */
        if (!anything_left_to_read) break;
    } while (1);

    /* reset - go back to 0 */
    fseek(fh, 0, SEEK_SET);

    /* return whether or not we have found an inconsistency */
    return !did_find_error;
}

/*
get number of dimensions in fh
-1 indicates inconsistent number of commas per line
0 indicates empty file (no line has meaningful characters)
>=1 indicates at least one point per line
*/
int get_number_of_dimensions(FILE* fh) {
    bool anything_left_to_read;
    int d;

    /* skip all newline characters */
    anything_left_to_read = FILE_skip_meaningless(fh);
    if (!anything_left_to_read) return 0;

    /* get dimensions in first ACTUAL line */
    d = get_number_of_dimensions_in_line(fh);

    /* reset - go back to 0 */
    fseek(fh, 0, SEEK_SET);

    if (is_number_of_dimensions_consistent(fh, d)) {
        return d;
    } else {
        return -1;
    }
}

int get_number_of_lines(FILE* fh) {
    int lines;
    char c;
    lines = 0;

    while (1) {
        while ((c = fgetc(fh)) == '\n'); /* skip over empty lines */
        if (c == EOF) break;             /* rollback, return if EOF */
        else lines++;                    /* we see something! */
        do { c = fgetc(fh); } while ((c != '\n') && (c != EOF));
    }

    fseek(fh, 0, SEEK_SET); /* so others can keep reading from the file */
    return lines;
}

status_t read_data(mat_t** dst, char* path_to_input) {
    int i, j, h, w;
    double num;
    file_read_enum num_read_result;
    FILE* fh;
    mat_t* x;

    *dst = NULL;

    fh = fopen(path_to_input, "rb");
    if (!fh) return ERROR_FOPEN;

    /* we assume that every line gets the same number of commas */
    w = get_number_of_dimensions(fh);
    h = get_number_of_lines(fh);
    if ((w <= 0) || (h <= 0)) {
        printd(("Could not fetch dimensions- (h,w)=(%d,%d)\n", h,w));
        fclose(fh);
        return ERROR_FORMAT; /* FIXME - does this yield "generic" or "invalid"? */
    }

    x = mat_init(h, w);
    if (!x) {
        printd(("cannot allocate!\n"));
        return ERROR_MALLOC;
    }
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            num_read_result = FILE_get_next_num(fh, &num);
            if (num_read_result != SUCCESS_VALID_CONTENT) {
                printd(("WEE WOO! We have a problem\n"));
                printd(("num @ index (%d, %d) is not parsed properly, result is %d\n", i, j, num_read_result));
                fclose(fh);
                mat_free(&x);
                return ERROR_FORMAT; /* FIXME - generic or invalid? */
            }
            mat_set(x, i, j, num);
        }
    }

    /* FIXME - more rigid format validation - verify that all lines have dimension w */

    fclose(fh);
    *dst = x;
    return SUCCESS;
}

status_t write_data(mat_t* src, char* path_to_output) {
    int i, j, h, w;
    FILE* fh;

    fh = fopen(path_to_output, "wb");
    if (!fh) printd(("cannot open\n"));
    if (!fh) return ERROR_FOPEN;

    /* we assume that every line gets the same number of commas */
    w = src->w;
    h = src->h;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            fprintf(fh, "%f", mat_get(src, i, j));
            if (j < (w-1)) fprintf(fh, ",");
        }
        fprintf(fh, "\n");
    }

    fclose(fh);
    return SUCCESS;
}
