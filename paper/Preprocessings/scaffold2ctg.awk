#!/usr/bin/awk -f

BEGIN {
    RS=">"
    FS="\n"
    gap_size=3
    contig_count=1
    output_fasta = "contigs.fasta"
    output_map = "scaffold_map.txt"
}

NR > 1 {
    scaffold_name = $1
    seq = ""
    for (i = 2; i <= NF; i++) {
        seq = seq $i
    }

    start = 1
    contig_list = ""  # liste des contigs pour ce scaffold

    for (i = 1; i <= length(seq) - gap_size + 1; i++) {
        if (substr(seq, i, gap_size) == "NNN") {
            contig = substr(seq, start, i - start)
            if (length(contig) > 0) {
                contig_name = "contig_" contig_count
                printf(">%s\n%s\n", contig_name, contig) >> output_fasta
                contig_list = contig_list contig_name " "
                contig_count++
            }
            start = i + gap_size
            i = start - 1
        }
    }

    # Dernier contig après le dernier "NNN"
    contig = substr(seq, start)
    if (length(contig) > 0) {
        contig_name = "contig_" contig_count
        printf(">%s\n%s\n", contig_name, contig) >> output_fasta
        contig_list = contig_list contig_name " "
        contig_count++
    }

    # Écriture dans le fichier de correspondance
    printf("%s\t%s\n", scaffold_name, contig_list) >> output_map
}
