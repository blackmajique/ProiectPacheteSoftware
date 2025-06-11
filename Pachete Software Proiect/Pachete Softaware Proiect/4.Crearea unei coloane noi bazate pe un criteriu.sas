data procesare;
    set students;
    length Performanta $15;
    if Final_Score >= 90.00 then Performanta = "Excelent";
    else if Final_Score >= 70.00 then Performanta = "Bun";
    else Performanta = "Slab";
run;
