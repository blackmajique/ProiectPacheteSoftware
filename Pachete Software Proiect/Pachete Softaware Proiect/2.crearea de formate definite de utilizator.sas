proc format;
    value $gender_fmt
        'M' = 'Masculin'
        'F' = 'Feminin';
    value grade_fmt
        low-40.00 = 'Sub Nivel'
        40.00-65.00 = 'Mediu'
        65.00-90.00 = 'Bine'
        90.00-high = 'Excelent';
run;
