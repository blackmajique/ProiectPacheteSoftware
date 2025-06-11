proc import datafile="/home/u64202692/Students_Grading_Dataset1.csv"
    out=studenti
    dbms=csv
    replace;
    delimiter=';';
    guessingrows=MAX;
run;

proc format;
    value $gender_fmt
        "Male" = "Barbat"
        "Female" = "Femeie";
    value $yn_fmt
        "Yes" = "Da"
        "No"  = "Nu";
    value $edu_fmt
        "High School" = "Liceu"
        "Bachelor's" = "Licenta"
        "Master's" = "Master"
        "PhD" = "Doctorat"
        "No formal education" = "Fara studii";
    value $income_fmt
        "Low" = "Venit scazut"
        "Medium" = "Venit mediu"
        "High" = "Venit ridicat";
run;

proc print data=studenti(obs=10);
    var Gender Extracurricular_Activities Internet_Access_at_Home 
        Parent_Education_Level Family_Income_Level;
    format Gender $gender_fmt.
           Extracurricular_Activities $yn_fmt.
           Internet_Access_at_Home $yn_fmt.
           Parent_Education_Level $edu_fmt.
           Family_Income_Level $income_fmt.;
    title "Date formatate - primele 10 randuri";
run;

proc freq data=studenti;
    tables Gender Extracurricular_Activities Internet_Access_at_Home 
           Parent_Education_Level Family_Income_Level;
    format Gender $gender_fmt.
           Extracurricular_Activities $yn_fmt.
           Internet_Access_at_Home $yn_fmt.
           Parent_Education_Level $edu_fmt.
           Family_Income_Level $income_fmt.;
    title "Frecvente formatate";
run;

data studenti_buni;
    set studenti;
    where Total_Score >= 70;
run;

data fara_internet;
    set studenti;
    where Internet_Access_at_Home = "No";
run;

data risc_educational;
    set studenti;
    where Family_Income_Level = "Low" and Total_Score < 60;
run;

data reusite_context_defavorizat;
    set studenti;
    where Parent_Education_Level = "None" and Total_Score >= 85;
run;

data studenti_eticheta;
    set studenti;
    length Tip_Student $15;
    if Participation_Score >= 70 and Extracurricular_Activities = "Yes" then
        Tip_Student = "Activ complet";
    else if Participation_Score >= 70 then
        Tip_Student = "Activ la curs";
    else if Extracurricular_Activities = "Yes" then
        Tip_Student = "Activ social";
    else
        Tip_Student = "Pasiv";
run;

data studenti_clasificati;
    set studenti;
    length Scor_Categorie $10;
    if Total_Score >= 90 then Scor_Categorie = "Ridicat";
    else if Total_Score >= 60 then Scor_Categorie = "Mediu";
    else Scor_Categorie = "Scăzut";
run;

data scoruri_bune;
    set studenti;
    array note[5] Midterm_Score Final_Score Projects_Score Quizzes_Avg Assignments_Avg;
    Nr_Componente_Bune = 0;

    do i = 1 to 5;
        if note[i] >= 70 then Nr_Componente_Bune + 1;
    end;

    drop i;
run;

data eticheta_student;
  set studenti;
  Nume_Complet = catx(" ", upcase(First_Name), upcase(Last_Name));
  Eticheta = catx(" - ", Nume_Complet, Family_Income_Level);
run;

data diferente_quiz;
  set studenti;
  Diferenta = abs(Total_Score - Quizzes_Avg);
run;

data medie_generala;
  set studenti;
  Medie_Componente = mean(Midterm_Score, Final_Score, Projects_Score, Quizzes_Avg, Assignments_Avg);
run;

data studenti_2023 studenti_2024;
    set studenti;
    if mod(_N_, 2) = 0 then output studenti_2023; 
    else output studenti_2024;                  
run;
data toti_studentii;
    set studenti_2023 studenti_2024;
run;

data academice;
  set studenti;
  keep Student_ID Midterm_Score Final_Score Total_Score Grade;
run;

data socio;
  set studenti;
  keep Student_ID Gender Family_Income_Level Parent_Education_Level;
run;

proc sort data=academice; by Student_ID; run;
proc sort data=socio; by Student_ID; run;

data studenti_combinati;
  merge academice socio;
  by Student_ID;
run;

data bursieri;
  set studenti;
  if Total_Score >= 85 then do;
    Tip_Bursa = "Merit";
    output;
  end;
run;

proc sql;
  create table studenti_bursieri as
  select s.Student_ID, s.First_Name, s.Last_Name, s.Total_Score, b.Tip_Bursa
  from studenti as s
  inner join bursieri as b
  on s.Student_ID = b.Student_ID;
quit;

proc sgplot data=students;
    vbar Gender / response=Total_Score stat=mean datalabel;
    yaxis label="Media Notei Finale";
    xaxis label="Gen";
run;

proc freq data=studenti noprint;
  tables Family_Income_Level / out=venituri_freq;
run;
proc gchart data=venituri_freq;
  pie Family_Income_Level / sumvar=Count value=inside percent=inside slice=outside;
  title "Distributia studentilor in functie de nivelul venitului familial";
run;

proc sgplot data=studenti;
  histogram Total_Score / binwidth=5;
  density Total_Score;
  title "Distributia scorurilor totale";
run;

proc sgplot data=studenti;
  vbox Total_Score / category=Gender;
  title "Compararea scorurilor în funcție de gen";
run;

proc means data=students mean std min max maxdec=2;
    var Final_Score Total_Score Quizzes_Avg Assignments_Avg Projects_Score;
run;

proc reg data=studenti;
  model Total_Score = Study_Hours_per_Week "Stress_Level (1-10)"n;
  title "Regresie liniara: efectul stresului si a studiului asupra scorului total";
run;

proc logistic data=studenti_logistic;
  class Family_Income_Level Internet_Access_at_Home / param=ref;
  model High_Performer(event='1') = Family_Income_Level Internet_Access_at_Home "Stress_Level (1-10)"n;
  title "Regresie logistica: probabilitatea unui scor final ≥ 85";
run;
























