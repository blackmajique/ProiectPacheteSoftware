proc import datafile="/home/u64202433/Pachete_Software_Proiect/Students_Grading_Dataset.csv"
    out=work.students
    dbms=csv
    replace;
    delimiter=';';
    getnames=yes;
run;
