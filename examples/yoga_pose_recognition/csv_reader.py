import csv
  
# open input CSV file as source
# open output CSV file as result
with open("./fitness_poses_csvs_out_processed/mountain.csv", "r") as source:
    reader = csv.reader(source)
      
    with open("./fitness_poses_csvs_out_processed _f/mountain.csv", "w") as result:
        writer = csv.writer(result)
        for r in reader:
            
            # Use CSV Index to remove a column from CSV
            #r[3] = r['year']
            tuple_cols = []
            tuple_cols = (r[0],r[1],r[2],r[5],r[6],r[11],r[12],r[15],r[16],r[17],r[18],r[23],r[24],r[25],r[26],r[27],r[28],r[29],r[30],r[31],r[32],r[33],r[34],r[47],r[48],r[49],r[50],r[51],r[52],r[53],r[54],r[55],r[56],r[57],r[58])

            writer.writerow(tuple_cols)