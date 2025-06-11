proc sgplot data=students;
    vbar Gender / response=Total_Score stat=mean datalabel;
    yaxis label="Media Notei Finale";
    xaxis label="Gen";
run;
