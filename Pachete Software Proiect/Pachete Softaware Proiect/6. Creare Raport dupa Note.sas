proc report data=students nowd;
    column First_Name Gender Final_Score;
    define Gender / format=$gender_fmt.;
    define Final_Score / format=grade_fmt.;
run;
