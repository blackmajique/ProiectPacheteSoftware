data students;
    set students;
    Medie_Evaluari = mean(Assignments_Avg, Quizzes_Avg, Projects_Score);
    Initiala_Prenume = substr(First_Name, 1, 1);
run;


proc print data=students (obs=10);
    var Student_ID First_Name Assignments_Avg Quizzes_Avg Projects_Score Medie_Evaluari Initiala_Prenume;
run;

