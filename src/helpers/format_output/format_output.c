//
// Created by User on 09.10.2026.
//

#include "format_output.h"

#include <stdarg.h>
#include <stdio.h>

void std_output_line(bool verbose, const char *text_line, ...) {
	if (!verbose) return;
	va_list args;
	va_start(args, text_line);
	vprintf(text_line, args);
	va_end(args);
}

void std_output_vector(char *text_line, bool verbose) {
}
