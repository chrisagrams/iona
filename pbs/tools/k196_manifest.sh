#!/bin/bash
# manifest of a tree: type, relpath, size, md5 (files) / link target (symlinks); plus stat of every symlink target
cd "$1" || exit 1
find . -mindepth 1 \( -type f -printf 'f\t%P\t%s\t' -exec ionice -c3 md5sum {} \; \) -o \( -type l -printf 'l\t%P\t%l\n' \) -o \( -type d -printf 'd\t%P\n' \) | awk -F'\t' 'BEGIN{OFS="\t"} $1=="f"{split($4,a," "); print $1,$2,$3,a[1]; next} {print}' | sort -t$'\t' -k2
find . -type l -printf '%P\n' | while read l; do t=$(readlink -f -- "$l"); [[ -e $t ]] && echo -e "T\t$l\t$(stat -L -c '%s %Y %i' -- "$l")" || echo -e "T\t$l\tDANGLING"; done | sort
