import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import vl_convert as vlc

numberOfDays = snakemake.config['stop'];

numberofPseudonyms = []
numberofPseudonymsBadOutcome = []
auprcRandomGuessing = []

combinationTest = []
combinationTraining = []

accuracies = {'Combination Test': [], 'Combination Training': [] }
precisions = {'Combination Test': [], 'Combination Training': [] }
recalls = {'Combination Test': [], 'Combination Training': [] }
matrix = {'Combination Test': [], 'Combination Training': [] }
matrix1 = {'Combination Test': [], 'Combination Training': [] }
matrix2 = {'Combination Test': [], 'Combination Training': [] }
matrix3 = {'Combination Test': [], 'Combination Training': [] }
f1scores = {'Combination Test': [], 'Combination Training': [] }
aurocScores = {'Combination Test': [], 'Combination Training': [] }
auprcScores = {'Combination Test': [], 'Combination Training': [] }

# read in data
j = 0;
for i in numberOfDays:
    both = pd.read_csv(snakemake.input[i], delimiter = ',');  # both (ts and no ts data)


    numberofPseudonyms.append((int(both['confusionMatrixTopLeftTest']) + int(both['confusionMatrixTopRightTest']) + int(both['confusionMatrixBottomLeftTest']) + int(both['confusionMatrixBottomRightTest'])+ int(both['confusionMatrixTopLeftTraining']) + int(both['confusionMatrixTopRightTraining']) + int(both['confusionMatrixBottomLeftTraining']) + int(both['confusionMatrixBottomRightTraining'])))
    numberofPseudonymsBadOutcome.append((int(both['confusionMatrixBottomLeftTest']) + int(both['confusionMatrixBottomRightTest']) + int(both['confusionMatrixBottomLeftTraining']) + int(both['confusionMatrixBottomRightTraining'])))
    auprcRandomGuessing.append(round((int(both['confusionMatrixBottomLeftTest']) + int(both['confusionMatrixBottomRightTest']))/(int(both['confusionMatrixTopLeftTest']) + int(both['confusionMatrixTopRightTest']) + int(both['confusionMatrixBottomLeftTest']) + int(both['confusionMatrixBottomRightTest'])), 6))

    # save combination data
    combinationTest.append([both['accuracyTest'], both['accuracyNotNormalizedTest'], both['confusionMatrixTopLeftTest'], both['confusionMatrixTopRightTest'], both['confusionMatrixBottomLeftTest'],both['confusionMatrixBottomRightTest'], both['recallTest'],both['precisionTest'], both['f1Test'], both['aurocTest'], both['auprcTest']])
    accuracies['Combination Test'].append(combinationTest[j][0])
    precisions['Combination Test'].append(combinationTest[j][7])
    recalls['Combination Test'].append(combinationTest[j][6])
    matrix['Combination Test'].append(combinationTest[j][2])
    matrix1['Combination Test'].append(combinationTest[j][3])
    matrix2['Combination Test'].append(combinationTest[j][4])
    matrix3['Combination Test'].append(combinationTest[j][5])
    f1scores['Combination Test'].append(combinationTest[j][8])
    aurocScores['Combination Test'].append(combinationTest[j][9])
    auprcScores['Combination Test'].append(combinationTest[j][10])


    combinationTraining.append([both['accuracyTraining'], both['accuracyNotNormalizedTraining'], both['confusionMatrixTopLeftTraining'], both['confusionMatrixTopRightTraining'], both['confusionMatrixBottomLeftTraining'],both['confusionMatrixBottomRightTraining'], both['recallTraining'],both['precisionTraining'], both['f1Training'], both['aurocTraining'], both['auprcTraining']])
    accuracies['Combination Training'].append(combinationTraining[j][0])
    precisions['Combination Training'].append(combinationTraining[j][7])
    recalls['Combination Training'].append(combinationTraining[j][6])
    matrix['Combination Training'].append(combinationTraining[j][2])
    matrix1['Combination Training'].append(combinationTraining[j][3])
    matrix2['Combination Training'].append(combinationTraining[j][4])
    matrix3['Combination Training'].append(combinationTraining[j][5])
    f1scores['Combination Training'].append(combinationTraining[j][8])
    aurocScores['Combination Training'].append(combinationTraining[j][9])
    auprcScores['Combination Training'].append(combinationTraining[j][10])

    j = j + 1;


# set title based on resistance wildcard
plotTitle = ''
if((snakemake.wildcards.resistance == 'resistance')):
    plotTitle = 'With Resistance Features'
elif((snakemake.wildcards.resistance == 'noResistance')):
    plotTitle = 'Without Resistance Features'



# plot the number of pseudonyms
fig, numbers = plt.subplots()
numbers.bar(numberOfDays, numberofPseudonyms)
numbers.set_ylim([810, 850])
numbers.set_ylabel('Number of Pseudonyms')
numbers.set_xlabel('Days')
plt.savefig(snakemake.output[0])

fig, numbers = plt.subplots()
numbers.bar(numberOfDays, numberofPseudonymsBadOutcome)
#numbers.set_ylim([820, 860])
numbers.set_ylabel('Number of Pseudonyms')
numbers.set_xlabel('Days')
plt.savefig(snakemake.output[10])



# f1 score
data7 = pd.DataFrame({
    "Days": numberOfDays,
    "F1": f1scores['Combination Test'],
    "name": "Both Test"})
data8 = pd.DataFrame({
    "Days": numberOfDays,
    "F1": f1scores['Combination Training'],
    "name": "Both Training"})

sourceF1 = pd.concat([data7, data8])

# F1 to table
sourceF1.to_csv(snakemake.output[19])


acc = alt.Chart(sourceF1).mark_bar().encode(
    alt.X("name:O", sort=['Both Test', 'Both Training']),
    y = "F1:Q",
    color = "name:N",
    column = "Days:O"
)
acc.save(snakemake.output[1])

# f1 line plot
srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = "Days:O",
    y = alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test'], scale = alt.Scale(range=['#88d27a']), legend=alt.Legend(title="Feature Set"))
)

srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = alt.X("Days:O", title = ''),
    y = alt.Y("F1:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N", sort=['Both Training'], scale = alt.Scale(range=['#54a24b']), legend=alt.Legend(title="Feature Set"))
)
vertical1 = alt.vconcat(line2, line1).resolve_scale(color = 'independent').configure_legend(titleFontSize = 13, labelFontSize = 12).configure_axis(titleFontSize=13).configure_title(fontSize = 15)

vertical1.save(snakemake.output[2])
vertical1.save(snakemake.output[3])


# data for AUROC and AUPRC
data7 = pd.DataFrame({
    "Days": numberOfDays,
    "AUROC": aurocScores['Combination Test'],
    "AUPRC": auprcScores['Combination Test'],
    "name": "Both Test"})
data8 = pd.DataFrame({
    "Days": numberOfDays,
    "AUROC": aurocScores['Combination Training'],
    "AUPRC": auprcScores['Combination Training'],
    "name": "Both Training"})

# plot AUROC
sourceF1 = pd.concat([data7, data8])

# AUROC and AUPRC to table
sourceF1.to_csv(snakemake.output[17])

acc = alt.Chart(sourceF1).mark_bar().encode(
    alt.X("name:O", sort=['Both Test', 'Both Training']),
    y = "AUROC:Q",
    color = "name:N",
    column = "Days:O"
)
acc.save(snakemake.output[4])


# AUROC line plot
srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = alt.X("Days:O"),
    y = alt.Y("AUROC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test'], scale = alt.Scale(range=['#88d27a']), legend=alt.Legend(title="Feature Set")),
)
srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = alt.X("Days:O", title = ''),
    y = alt.Y("AUROC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Training'], scale = alt.Scale(range=['#54a24b']), legend=alt.Legend(title="Feature Set")),
)
vertical1 = alt.vconcat(line2, line1).resolve_scale(color = 'independent').configure_legend(titleFontSize = 13, labelFontSize = 12).configure_axis(titleFontSize=13).configure_title(fontSize = 15)

vertical1.save(snakemake.output[5])
vertical1.save(snakemake.output[6])

# plot AUROC line plot with the number of pseudonyms bar chart
srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUROC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test'], scale = alt.Scale(range=['#88d27a']), title = 'Model')
)

srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUROC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Training'], title= 'Model', scale = alt.Scale(range=['#54a24b']))
)

dataBarChart = pd.DataFrame({
    "Days": numberOfDays,
    "Pseudonyms": numberofPseudonymsBadOutcome})

pseudonymsBarChart = alt.Chart(dataBarChart).mark_bar(color='#d9d9d9', clip=True).encode(
    x = "Days:O",
    y = alt.Y("Pseudonyms:Q", scale=alt.Scale(domain=[50, 85])),
)

layer1 = alt.layer(pseudonymsBarChart, line1).resolve_scale(
    y='independent'
)

layer2 = alt.layer(pseudonymsBarChart, line2).resolve_scale(
    y='independent'
)

completeChart = alt.vconcat(layer2, layer1).resolve_scale(color = 'independent')


completeChart.save(snakemake.output[11])
completeChart.save(snakemake.output[12])


# plot AUPRC

sourceF1 = pd.concat([data7, data8])

acc = alt.Chart(sourceF1).mark_bar().encode(
    alt.X("name:O", sort=['Both Test', 'Both Training']),
    y = "AUPRC:Q",
    color = "name:N",
    column = "Days:O"
)
acc.save(snakemake.output[7])

# AUPRC line plot
srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test'], scale = alt.Scale(range=['#88d27a'])),
)

srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Training'], scale = alt.Scale(range=['#54a24b'])),
)
vertical1 = alt.vconcat(line2, line1).resolve_scale(color = 'independent')

vertical1.save(snakemake.output[8])
vertical1.save(snakemake.output[9])

# AUPRC line plot with the number of pseudonyms bar chart
srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test'], scale = alt.Scale(range=['#88d27a']))
)

srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Training'], scale = alt.Scale(range=['#54a24b']))
)

dataBarChart = pd.DataFrame({
    "Days": numberOfDays,
    "Pseudonyms": numberofPseudonymsBadOutcome})

pseudonymsBarChart = alt.Chart(dataBarChart).mark_bar(color='#d9d9d9', clip=True).encode(
    x = "Days:O",
    y = alt.Y("Pseudonyms:Q", scale=alt.Scale(domain=[50, 85])),
)

layer1 = alt.layer(pseudonymsBarChart, line1).resolve_scale(
    y='independent'
)

layer2 = alt.layer(pseudonymsBarChart, line2).resolve_scale(
    y='independent'
)

completeChart = alt.vconcat(layer2, layer1).resolve_scale(color = 'independent')


completeChart.save(snakemake.output[13])
completeChart.save(snakemake.output[14])


# AUPRC line plot with random guessing line
rd = pd.DataFrame({
    "Days": numberOfDays,
    "AUPRC": auprcRandomGuessing,
    "name": 'Random Guessing'})

# save random guessing to table
rd.to_csv(snakemake.output[18])

srcLine1 = pd.concat([data7])
line1 = alt.Chart(srcLine1).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test', 'Random Guessing'], scale = alt.Scale(range=['#88d27a', 'red']), legend=alt.Legend(title="Feature Set")),
)

line3 = alt.Chart(rd).mark_line().encode(
    x = "Days:O",
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Test', 'Random Guessing'], scale = alt.Scale(range=['#88d27a', 'red']), legend=alt.Legend(title="Feature Set"))
)


srcLine2 = pd.concat([data8])
line2 = alt.Chart(srcLine2, title = plotTitle).mark_line().encode(
    x = alt.X("Days:O", title = ''),
    y = alt.Y("AUPRC:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N",sort=['Both Training'], scale = alt.Scale(range=['#54a24b']), legend=alt.Legend(title="Feature Set")),

)

vertical1 = alt.vconcat(line2, alt.layer(line3, line1)).resolve_scale(color = 'independent').configure_legend(titleFontSize = 13, labelFontSize = 12).configure_axis(titleFontSize=13).configure_title(fontSize = 15)

vertical1.save(snakemake.output[15])
vertical1.save(snakemake.output[16])


#### Paper Figure 3 ####

rg = rd.rename(columns={
    'AUPRC' : 'RGAUPRC'
})
auroc_auprc = pd.concat([data7, data8])
auroc_auprc = auroc_auprc[auroc_auprc['name'] == 'Both Test']

combined = pd.merge(rg,auroc_auprc,on='Days')[['Days','RGAUPRC','AUROC','AUPRC']]
combined['AUROC'] = combined['AUROC'].apply( lambda  x : x.values[0])
combined['AUPRC'] = combined['AUPRC'].apply( lambda x : x.values[0])
combined['RGAUROC'] = 0.5

combined = combined.melt(id_vars='Days',var_name='Score',value_name='Value')

#AUROC
auroc = combined[combined['Score'].isin(['RGAUROC','AUROC'])]
auroc['Method'] = auroc['Score'].map({'RGAUROC' : 'Random Guessing', 'AUROC' : 'Model'})

c1= alt.Chart(auroc).mark_line(point=True).encode(
    x='Days',
    y=alt.Y('Value',title='AUROC'),
    color='Method'
)

#AUPRC
auprc = combined[combined['Score'].isin(['RGAUPRC','AUPRC'])]
auprc['Method'] = auprc['Score'].map({'RGAUPRC' : 'Random Guessing', 'AUPRC' : 'Model'})

c2= alt.Chart(auprc).mark_line(point=True).encode(
    x='Days',
    y=alt.Y('Value',title='AUPRC'),
    color='Method'
)

perform = (c1|c2)
perform.save(snakemake.output[20])
