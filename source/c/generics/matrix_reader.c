#include "matrix_reader.h"

int FILE_locate(FILE* fh, char needle) {
    char c;
    int counted;
    long start;
    start = ftell(fh);
    counted = 0;
    while (1) {
        c = fgetc(fh);
        if ((c == EOF) || (c == needle)) {
            break;
        }
        counted++;
    }
    fseek(fh, start, SEEK_SET);

    if (c != needle) return -1;
    else return counted;
}

char* FILE_get(FILE* fh, int length) {
    char* line;
    if (length < 0) length = 0;
    line = malloc(/*sizeof(char) * */(length + 1));
    if (line == NULL) return NULL;

    fgets(line, length+1, fh);

    return line;
}

char* FILE_get_next_line(FILE* fh) {
    char* line;
    int length = FILE_locate(fh, '\n');
    if (length < 0) length = 0;
    line = FILE_get(fh, length);
    if ((line != NULL) && (!feof(fh))) fgetc(fh); /* drop last */
    return line;
}

char* FILE_get_next_num(FILE* fh) {
    int next_comma, next_newl;
    char* line;

    next_comma = FILE_locate(fh, ',');
    next_newl  = FILE_locate(fh, '\n');
    if ((next_comma == -1) || ((next_newl != -1) && (next_newl < next_comma))) {
        next_comma = next_newl;
    }

    if (next_comma < 0) next_comma = 0;


    line = FILE_get(fh, next_comma);

    if ((line != NULL) && (!feof(fh))) fgetc(fh); /* drop last */

    return line;
}

int get_number_of_dimensions(FILE* fh) {
    int i, d, got_dim;
    char* line;
    d = 1;
    got_dim = 0;
    while (!feof(fh)) {
        line = FILE_get_next_line(fh);
        if ((line[0] == 0) || (line[0] == '\n') || (line[0] == ',')) {
            free(line);
            continue;
        }
        got_dim = 1;
        for (i = 0; (line[i] != 0) && (line[i] != '\n'); i++) {
            if (line[i] == ',') d++;
        }
        free(line);
        break;
    }
    if (!got_dim) d = 0;
    fseek(fh, 0, SEEK_SET);
    return d;
}

int get_number_of_lines(FILE* fh) {
    int lines;
    char c;
    lines = 0;
    while (1) {
        c = fgetc(fh);
        if (c == EOF) break;
        if (c == '\n') lines++;
    }
    fseek(fh, 0, SEEK_SET);
    return lines;
}

status_t read_data(mat_t** dst, char* path_to_input) {
    int i, j, h, w;
    char* line;
    FILE* fh;
    mat_t* x;

    *dst = NULL;

    fh = fopen(path_to_input, "rb");
    if (!fh) return ERROR_FOPEN;

    /* we assume that every line gets the same number of commas */
    w = get_number_of_dimensions(fh);
    h = get_number_of_lines(fh);
    if ((w <= 0) || (h <= 0)) {
        fclose(fh);
        return ERROR_FORMAT;;
    }

    x = mat_init(h, w);
    if (!x) return ERROR_MALLOC;
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            line = FILE_get_next_num(fh);
            mat_set(x, i, j, atof(line));
            free(line);
        }
    }

    /* FIXME - more rigid format validation - verify that all lines have dimension w */

    fclose(fh);
    *dst = x;
    return SUCCESS;
}