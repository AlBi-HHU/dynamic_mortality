import pandas as pd
import numpy as np
import altair as alt

numberOfDays = snakemake.config['stop'];

# binary outcomes
def set_outcome(row):
    if(isinstance(row['Date'], str) == True):
        return '0-100'
    else: # else is NaN, which is of type float
        return 'Alive'

# split bad outcomes into two groups
def set_timeframe(row):
    if(isinstance(row['Date'], str) == True):
        day = int(row['Date'])
        if(day <= 30):
            return '0-30'
        else:
            return '31-100'
    else:
        return 'Alive'

# split bad outcomes into more smaller groups
def set_detailed_timeframe(row):
    if(isinstance(row['Date'], str) == True):
        day = int(row['Date'])
        if(day <= 30):
            return '0-30'
        elif(day <= 50):
            return '31-50'
        elif(day <= 75):
            return '51-75'
        else:
            return '76-100'
    else:
        return 'Alive'


# set title based on ts and resistance wildcard
plotTitle = ''
if((snakemake.wildcards.ts == 'both') and (snakemake.wildcards.resistance == 'resistance')):
    plotTitle = 'All Features'
elif((snakemake.wildcards.ts == 'both') and (snakemake.wildcards.resistance == 'noResistance')):
    plotTitle = 'Without Resistance Features'
elif((snakemake.wildcards.ts == 'ts') and (snakemake.wildcards.resistance == 'resistance')):
    plotTitle = 'Without Clinical Measurements'
elif((snakemake.wildcards.ts == 'ts') and (snakemake.wildcards.resistance == 'noResistance')):
    plotTitle = 'Without Clinical Measurements and Resistance Features'
elif((snakemake.wildcards.ts == 'noTs') and (snakemake.wildcards.resistance == 'resistance')):
    plotTitle = 'Without Time Series Features'
elif((snakemake.wildcards.ts == 'noTs') and (snakemake.wildcards.resistance == 'noResistance')):
    plotTitle = 'Without Time Series and Resistance Features'

plotTitle = 'Test: ' + plotTitle


# read in data
complete = []
allThresholds = []
j = 0

highLow = pd.read_csv(snakemake.input[63], delimiter= ',')
highLowQ = pd.read_csv(snakemake.input[64], delimiter= ',')
highLowT = pd.read_csv(snakemake.input[65], delimiter= ',')


for i in numberOfDays:
    lowOne = highLow.at[i, 'low']
    highOne = highLow.at[i, 'high']

    lowOneQ = highLowQ.at[i, 'low']
    highOneQ = highLowQ.at[i, 'high']

    lowOneT = highLowT.at[i, 'low']
    highOneT = highLowT.at[i, 'high']

    data = pd.read_csv(snakemake.input[j], delimiter= ',').astype(str)
    data.drop(columns = ['Unnamed: 0'], inplace = True)
    filler = np.full(len(data), i)
    data.insert(0, "day", filler)
    data.insert(0, "x", np.full(len(data), (i-0.5)))
    data.insert(0, "x2", np.full(len(data), (i+0.5)))

    lowHigh = []
    quantiles = []
    lowHighT = []
    for index, row in data.iterrows():
        prob = float(row['probabilities'])
        if(prob <= lowOne):
            lowHigh.append('low')
        elif(prob >= highOne):
            lowHigh.append('high')
        else:
            lowHigh.append('moderate')

        if(prob <= lowOneQ):
            quantiles.append('low')
        elif(prob >= highOneQ):
            quantiles.append('high')
        else:
            quantiles.append('moderate')

        if(prob <= lowOneT):
            lowHighT.append('low')
        elif(prob >= highOneT):
            lowHighT.append('high')
        else:
            lowHighT.append('moderate')

    data.insert(0, "lowHigh", lowHigh)
    data.insert(0, "quantiles", quantiles)
    data.insert(0, "lowHighThresholds", lowHighT)

    complete.append(data)



    thresholds = pd.read_csv(snakemake.input[j+31], delimiter = ',');
    index_threshold = -1
    t = 0
    for index, row in thresholds.iterrows():
        if(row['FPR'] <= 0.2):
            index_threshold = index
            t = row['Thresholds']
    allThresholds.append(t)

    j = j + 1;

thr_dict = {'day': np.arange(31) , 'threshold': allThresholds, 'name': 'Threshold', 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1)}
dfThresholds = pd.DataFrame(data = thr_dict)

outcomes = pd.read_csv(snakemake.input[62], delimiter= ';').astype(str)

df1 = pd.concat(complete)


df = df1.merge(outcomes,how='left', left_on='pseudonyms', right_on='Pseudonym')
df['outcome'] = df.apply(set_outcome, axis=1)
df['outcomeDay'] = df.apply(set_timeframe, axis=1)
df['detailedOutcomeDay'] = df.apply(set_detailed_timeframe, axis = 1)


pseudonyms = df['pseudonyms'][df.day.isin([0])]

keep = []
for p in pseudonyms:
    dataP = df[df.pseudonyms.isin([p])]
    bin = dataP['binary']
    zeros = bin.isin(['0'])
    ones = bin.isin(['1'])
    if(zeros.all() != True and ones.all() != True):
        keep.append(p)


# print pseudonyms that switch between binary outcomes
filteredData = df[df.pseudonyms.isin(keep)]
chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)


chart.save(snakemake.output[0])

chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[1])


## Print pseudonyms, that do not switch between binary outcomes
filteredData = df[~df.pseudonyms.isin(keep)]

chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)


chart.save(snakemake.output[2])

chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[3])


## Print all pseudonyms
chart = alt.Chart(df).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)


chart.save(snakemake.output[4])

chart = alt.Chart(df).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[5])


## Print all pseudonyms using outcomes as color
chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcome:O',sort = ['Alive', '0-100']),
    alt.Color("outcome:O",sort = ['Alive', '0-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518']))).properties(
    width=1200,
    height=1200)
chart.save(snakemake.output[6])

# Print all pseudonyms using three colours
chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    alt.Color("outcomeDay:O",sort = ['Alive','0-30', '31-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518','#54a24b']))).properties(
    width=1200,
    height=1200)
chart.save(snakemake.output[10])


# Print all pseudonyms using more colours
chart = alt.Chart(df).mark_point().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('detailedOutcomeDay:O',sort = ['Alive', '0-30', '31-50', '51-75', '76-100']),
    alt.Color("detailedOutcomeDay:O",sort = ['Alive','0-30', '31-50', '51-75', '76-100'], scale = alt.Scale(range = ['#4c78a8','#e41a1c', '#ffbf79',  '#d67195', '#54a24b'])).title('Outcome')).properties(
    width=2000,
    height=1000)
chart.save(snakemake.output[19])

# Print all pseudonyms using three colours + threshold
dots = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    alt.Color("outcomeDay:O",sort = ['Alive','0-30', '31-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518','#54a24b']))).properties(
    width=1200,
    height=1200)

line = alt.Chart(dfThresholds).mark_line().encode(
    x = "day:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2']))
)
chart = alt.layer(line, dots).resolve_scale(color = 'independent')

chart.save(snakemake.output[11])

# Low High Moderate
dots = alt.Chart(df).mark_point().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color("lowHigh:O",sort = ['high', 'moderate', 'low'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Risk')).properties(
    width=2000,
    height=1000)
line = alt.Chart(dfThresholds).mark_line().encode(
    x = "day:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2']))
)
chart = alt.layer(line, dots).resolve_scale(color = 'independent')
chart.save(snakemake.output[12])

# Low High Moderate Quantiles

dots = alt.Chart(df).mark_point().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=1750,
    height=800)

line = alt.Chart(dfThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))


highRisk = alt.Chart(highLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(highLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)
chart = alt.layer(line, lowRisk, highRisk, dots).resolve_scale(color = 'independent', y="shared")
chart.save(snakemake.output[13])


# Low High Moderate Thresholds 
dots = alt.Chart(df, title = plotTitle).mark_point().encode(
    alt.X("day:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])).title('Days'),
    alt.Y("probabilities:Q", scale=alt.Scale(domain=[0, 1])).title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive','31-100', '0-30']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=1750,
    height=800)

highRisk = alt.Chart(highLowT).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("high:Q",scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(highLowT).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("low:Q",scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)
chart = alt.layer(lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared", x = "shared").configure_axisX(
titleFontSize=20, labelFontSize = 17, titlePadding = 15, values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]).configure_axisY(titleFontSize = 20, labelFontSize = 17, titlePadding = 20).configure_title(fontSize = 22).configure_legend(titleFontSize = 20, labelFontSize = 17, symbolSize = 220, symbolStrokeWidth = 2)
chart.save(snakemake.output[20])
chart.save(snakemake.output[23])


# Low High Moderate Thresholds More Groups
dots = alt.Chart(df, title = plotTitle).mark_point().encode(
    alt.X("day:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])).title('Days'),
    alt.Y("probabilities:Q", scale=alt.Scale(domain=[0, 1])).title('Probability'),
    alt.XOffset('detailedOutcomeDay:O',sort = ['Alive','76-100', '51-75', '31-50', '0-30'], scale = alt.Scale(range= [0, 45])),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color("detailedOutcomeDay:O",sort = ['Alive','0-30', '31-50', '51-75', '76-100'], scale = alt.Scale(range = ['#4c78a8','#e41a1c', '#ffbf79',  '#d67195', '#54a24b'])).title('Outcome')).properties(
    width=2000,
    height=800)

highRisk = alt.Chart(highLowT).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("high:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(highLowT).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("low:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)
chart = alt.layer(lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared", x = "shared").configure_axisX(
titleFontSize=20, labelFontSize = 17, titlePadding = 15, values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]).configure_axisY(titleFontSize = 20, labelFontSize = 17, titlePadding = 20).configure_title(fontSize = 22).configure_legend(titleFontSize = 20, labelFontSize = 17, symbolSize = 220, symbolStrokeWidth = 2)
chart.save(snakemake.output[21])
chart.save(snakemake.output[24])


#### Boxplots
dots = alt.Chart(df).mark_boxplot(extent="min-max").encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=2000,
    height=1000)


line = alt.Chart(dfThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))


highRisk = alt.Chart(highLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(highLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)
chart = alt.layer(line, lowRisk, highRisk, dots).resolve_scale(color = 'independent', y="shared")
chart.save(snakemake.output[14])


# Low High Mid Quantiles On Day 1
filtered = df[df.day == 1]
dots = alt.Chart(filtered).mark_point().encode(
    alt.X("day:O", axis=alt.Axis(tickMinStep=1)).title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=250,
    height=350)
filteredThresholds = dfThresholds[dfThresholds.day == 1]
line = alt.Chart(filteredThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))

filteredHighLowQ = highLowQ[highLowQ.day == 1]

highRisk = alt.Chart(filteredHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(filteredHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(line,lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared").configure_axisX(tickMinStep=1.0)
chart.save(snakemake.output[17])

## Day 28
filtered = df[df.day == 28]
dots = alt.Chart(filtered).mark_point().encode(
    alt.X("day:O", axis=alt.Axis(tickMinStep=1)).title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=250,
    height=350)
filteredThresholds = dfThresholds[dfThresholds.day == 28]
line = alt.Chart(filteredThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))
filteredHighLowQ = highLowQ[highLowQ.day == 28]

highRisk = alt.Chart(filteredHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(filteredHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(line,lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared").configure_axisX(tickMinStep=1.0)
chart.save(snakemake.output[18])



# Low High Mid Thresholds On Day 1
filtered = df[df.day == 1]
dots = alt.Chart(filtered).mark_point().encode(
    alt.X("day:O", axis=alt.Axis(tickMinStep=1)).title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '31-100', '0-30']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=250,
    height=350)
filteredThresholds = dfThresholds[dfThresholds.day == 1]
line = alt.Chart(filteredThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))

filteredHighLowT = highLowT[highLowT.day == 1]

highRisk = alt.Chart(filteredHighLowT).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(filteredHighLowT).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared").configure_axisX(tickMinStep=1.0)
chart.save(snakemake.output[22])

#####################################

chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("outcome:O",sort = ['Alive', '0-100'],scale = alt.Scale(range = ['#4c78a8', '#f58518']))).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[7])


## Print all pseudonyms with bad outcomes
filteredData = df[df.outcome == '0-100']
chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)


chart.save(snakemake.output[8])

chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[9])

## Print all pseudonyms with good outcomes
filteredData = df[df.outcome == 'Alive']
chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)
chart.save(snakemake.output[26])


########## risk calculations
lB = []
mB = []
hB = []

f_lB = []
f_mB = []
f_hB = []

tB = []

c_lB = 0
c_mB = 0
c_hB = 0


lG = []
mG = []
hG = []

f_lG = []
f_mG = []
f_hG = []

tG = []

c_lG = 0
c_mG = 0
c_hG = 0

day = 0
for index, row in df.iterrows():
    if(day != int(row['day'])):
        day = day + 1
        lB.append(c_lB)
        mB.append(c_mB)
        hB.append(c_hB)
        t = (c_lB + c_mB + c_hB)
        tB.append(t)
        f_lB.append((c_lB/t))
        f_mB.append((c_mB/t))
        f_hB.append((c_hB/t))

        c_lB = 0
        c_mB = 0
        c_hB = 0


        lG.append(c_lG)
        mG.append(c_mG)
        hG.append(c_hG)
        t = (c_lG + c_mG + c_hG)
        tG.append(t)
        f_lG.append((c_lG/t))
        f_mG.append((c_mG/t))
        f_hG.append((c_hG/t))

        c_lG = 0
        c_mG = 0
        c_hG = 0

    """
    if(row['outcome'] == '0-100'):
        if(row['quantiles'] == 'low'):
            c_lB = c_lB +1
        elif(row['quantiles'] == 'moderate'):
            c_mB = c_mB +1
        elif(row['quantiles'] == 'high'):
            c_hB = c_hB +1
    elif(row['outcome'] == 'Alive'):
        if(row['quantiles'] == 'low'):
            c_lG = c_lG +1
        elif(row['quantiles'] == 'moderate'):
            c_mG = c_mG +1
        elif(row['quantiles'] == 'high'):
            c_hG = c_hG +1
    """

    if(row['outcome'] == '0-100'):
        if(row['lowHighThresholds'] == 'low'):
            c_lB = c_lB +1
        elif(row['lowHighThresholds'] == 'moderate'):
            c_mB = c_mB +1
        elif(row['lowHighThresholds'] == 'high'):
            c_hB = c_hB +1
    elif(row['outcome'] == 'Alive'):
        if(row['lowHighThresholds'] == 'low'):
            c_lG = c_lG +1
        elif(row['lowHighThresholds'] == 'moderate'):
            c_mG = c_mG +1
        elif(row['lowHighThresholds'] == 'high'):
            c_hG = c_hG +1

lB.append(c_lB)
mB.append(c_mB)
hB.append(c_hB)
t = (c_lB + c_mB + c_hB)
tB.append(t)
f_lB.append((c_lB/t))
f_mB.append((c_mB/t))
f_hB.append((c_hB/t))

lG.append(c_lG)
mG.append(c_mG)
hG.append(c_hG)
t = (c_lG + c_mG + c_hG)
tG.append(t)
f_lG.append((c_lG/t))
f_mG.append((c_mG/t))
f_hG.append((c_hG/t))

dict = {'day': np.arange(31), 'Low Risk 0-100 Absolute': lB, 'Low Risk 0-100': f_lB, 'Moderate Risk 0-100 Absolute': mB,'Moderate Risk 0-100': f_mB, 'High Risk 0-100 Absolute': hB, 'High Risk 0-100': f_hB, 'Total 0-100 Absolute': tB, 'Low Risk Alive Absolute': lG,'Low Risk Alive': f_lG, 'Moderate Risk Alive Absolute': mG,'Moderate Risk Alive': f_mG, 'High Risk Alive Absolute': hG, 'High Risk Alive': f_hG,'Total Alive Absolute': tG}
df_wsh = pd.DataFrame(data = dict)

df_wsh.to_csv(snakemake.output[15], sep = ',')
altDf_wsh_B = pd.melt(df_wsh,id_vars=['day'], value_vars=['Low Risk 0-100', 'Moderate Risk 0-100', 'High Risk 0-100'])
altDf_wsh_G = pd.melt(df_wsh,id_vars=['day'], value_vars=['Low Risk Alive', 'Moderate Risk Alive', 'High Risk Alive'])

chart = alt.Chart(altDf_wsh_B, title = plotTitle).mark_bar().encode(
    alt.X("day:O").title(None),
    alt.Y("sum(value)").title('# Patients (relative)'),
    alt.Color("variable:N", sort = ['Low Risk 0-100', 'Moderate Risk 0-100', 'High Risk 0-100']).title('0-100'),
    order=alt.Order('color_variable_sort_index:Q')
    ).properties(
    width=1000,
    height=300)
chart2 = alt.Chart(altDf_wsh_G).mark_bar().encode(
    alt.X("day:O").title('Days'),
    alt.Y("sum(value)").title('# Patients (relative)'),
    alt.Color("variable:N", sort = ['Low Risk Alive', 'Moderate Risk Alive', 'High Risk Alive']).title('Alive'),
    order=alt.Order('color_variable_sort_index:Q')
    ).properties(
    width=1000,
    height=300)
riskChart = alt.vconcat(chart, chart2).resolve_scale(color = 'independent').configure_axisX(titleFontSize=16, labelFontSize = 14, titlePadding = 10).configure_axisY(titleFontSize = 16, labelFontSize = 14, titlePadding = 15).configure_title(fontSize = 18).configure_legend(titleFontSize = 16, labelFontSize = 14, symbolSize = 200)
riskChart.save(snakemake.output[16])
riskChart.save(snakemake.output[25])


df.drop(['quantiles', 'lowHigh'], inplace= True, axis = 1)
df.to_csv(snakemake.output[27])



##### Paper Figures #####

test_data = df.astype({'pseudonyms': 'int64'})
test_data['semiDetailedOutcomeDay'] = test_data['detailedOutcomeDay'].apply(lambda x : x if x not in ['0-30','31-50'] else '0-50')

charts = []
numbers = [353, 568, 700, 153, 530, 482, 176, 431, 8, 483, 814]

## Supplementary Figure 1 ##

supplementary = alt.Chart(test_data).mark_rect().encode(
        x=alt.X('day:O',title='Day of Prediction'),
        y=alt.Y('pseudonyms:O',title=None),
        color=alt.Color(
            'lowHighThresholds',
            scale=alt.Scale(domain=['low','moderate','high'],
                            range=['green','orange','red']),
            title='Risk'
        )
    ).facet(row='semiDetailedOutcomeDay').resolve_scale(y='independent').configure_scale(
    bandPaddingInner=0.1
)
supplementary.save(snakemake.output[29])

## Figure 5 ##

fig = alt.Chart(test_data[test_data['pseudonyms'].isin(numbers)]).mark_rect().encode(
        x=alt.X('day:O',title='Day of Prediction'),
        y=alt.Y('pseudonyms:O',title=None),
        color=alt.Color(
            'lowHighThresholds',
            scale=alt.Scale(domain=['low','moderate','high'],
                            range=['green','orange','red']),
            title='Risk'
        )
    ).facet(row='semiDetailedOutcomeDay').resolve_scale(y='independent').configure_scale(
    bandPaddingInner=0.1
)
fig.save(snakemake.output[28])


## Figure 2 ##

fractions_groups_2 = (
    test_data.groupby(['outcome','lowHighThresholds','day'])['pseudonyms'].nunique()/
    test_data.groupby(['outcome','day'])['pseudonyms'].nunique()
).reset_index()

fractions_groups_2['order'] = fractions_groups_2['lowHighThresholds'].map({
    'low' : 0,
    'moderate' : 1,
    'high' : 2
})

def get_barplot(outcome):
    return alt.Chart(
    fractions_groups_2[
        fractions_groups_2['outcome'] == outcome
    ],title=outcome,
        height=200
).mark_bar().encode(
        x=alt.X('day:O',title='Day of Prediction'),
        y=alt.Y('pseudonyms',title='Fraction of Patients'),
        color=alt.Color(
            'lowHighThresholds',
            scale=alt.Scale(domain=['low','moderate','high'],
                            range=['green','orange','red']),
            title='Risk',
            sort=['low','moderate','high']
        ),
        order=alt.Order('order')
    )

alive = get_barplot('Alive')
d100 = get_barplot('0-100')

chart2= alt.Chart(test_data[
    (test_data['outcome'] == '0-100')&
    (test_data['lowHighThresholds'] == 'low')
],height=400).mark_bar().encode(
    x=alt.X('Date',bin=alt.Bin(maxbins=20)),
    color=alt.Color(
            'lowHighThresholds',
            scale=alt.Scale(domain=['low','moderate','high'],
                            range=['green','orange','red']),
            title='Risk',
            sort=['low','moderate','high'],
        legend=alt.Legend(orient='top')
        ),
    y='count()'
)

c = (alive&d100)|chart2
c.save(snakemake.output[30])



## Tables ##

fractions_groups_2 = (
    test_data.groupby(['outcome','lowHighThresholds','day'])['pseudonyms'].nunique()/
    test_data.groupby(['outcome','day'])['pseudonyms'].nunique()
).reset_index()

fractions_groups_2['order'] = fractions_groups_2['lowHighThresholds'].map({
    'low' : 0,
    'moderate' : 1,
    'high' : 2
})

table = fractions_groups_2.groupby(['outcome','lowHighThresholds'])['pseudonyms'].mean()
table.to_csv(snakemake.output[31])

total_per_outcome = test_data.groupby('outcome')['pseudonyms'].nunique().to_dict()

result = test_data.groupby(['outcome','lowHighThresholds'])['pseudonyms'].nunique().reset_index()
result['fraction'] = result.apply(lambda row: row['pseudonyms'] / total_per_outcome[row['outcome']], axis=1)
result.to_csv(snakemake.output[32])
