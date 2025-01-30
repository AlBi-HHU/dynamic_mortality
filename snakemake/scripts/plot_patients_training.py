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

plotTitle = 'Training: ' + plotTitle


complete = []
allThresholds = []
lowThresholds = []
highThresholds = []
minima = []
maxima = []
span = []
high = []
low = []
lowQ = []
highQ = []
j = 0

# read in data
for i in numberOfDays:

    data = pd.read_csv(snakemake.input[j], delimiter= ',').astype(str)
    data.drop(columns = ['Unnamed: 0'], inplace = True)
    filler = np.full(len(data), i)
    data.insert(0, "day", filler)
    data.insert(0, "x", np.full(len(data), (i-0.5)))
    data.insert(0, "x2", np.full(len(data), (i+0.5)))
    data['probabilities_float'] = data['probabilities'].astype(float)


    q = data.quantile([0.25, 0.5, 0.75], numeric_only = True, interpolation = 'linear')
    lowQuantileOne = float(q.at[0.25, 'probabilities_float'])
    highQuantileOne = float(q.at[0.75, 'probabilities_float'])

    lowQ.append(lowQuantileOne)
    highQ.append(highQuantileOne)

    min = 1
    max = 0
    for index, row in data.iterrows():
        prob = row['probabilities_float']
        if(prob < min):
            min = prob
        if(prob > max):
            max = prob
    minima.append(min)
    maxima.append(max)
    spanOne = (float(max)-float(min))
    lowOne = (min + (spanOne * 0.25))
    highOne = ((max - (spanOne * 0.25)))
    span.append(spanOne)
    high.append(highOne)
    low.append(lowOne)

    lowHigh = []
    quantiles = []

    l = 0
    h = 0
    m = 0

    for index, row in data.iterrows():
        prob = row['probabilities_float']
        if(prob <= lowOne):
            lowHigh.append('low')
        elif(prob >= highOne):
            lowHigh.append('high')
        else:
            lowHigh.append('moderate')

        if(prob <= lowQuantileOne):
            quantiles.append('low')
            l = l + 1
        elif(prob >= highQuantileOne):
            quantiles.append('high')
            h = h + 1
        else:
            quantiles.append('moderate')
            m = m + 1
    data.insert(0, "lowHigh", lowHigh)
    data.insert(0, "quantiles", quantiles)
    complete.append(data)


    # read test thresholds
    thresholds = pd.read_csv(snakemake.input[j+31], delimiter = ',');
    index_threshold = -1
    t = 0
    for index, row in thresholds.iterrows():
        if(row['FPR'] <= 0.2):
            index_threshold = index
            t = row['Thresholds']
    allThresholds.append(t)

    # read training thresholds
    thresholdsTraining = pd.read_csv(snakemake.input[j+62], delimiter = ',');
    index_threshold_high = -1
    index_threshold_low = -1
    h = 0
    l = 0
    for index, row in thresholdsTraining.iterrows():
        if(row['FPR'] <= 0.2):
            index_threshold_high = index
            h = row['Thresholds']
        if((row['TPR'] >= 0.95)):
            if(index_threshold_low == -1):
                index_threshold_low = index
                l = row['Thresholds']
            elif((index_threshold_low != -1) and (row['TPR'] == 0.95)):
                index_threshold_low = index
                l = row['Thresholds']
    lowThresholds.append(l)
    highThresholds.append(h)
    j = j + 1;

# check that all low risk thresholds are smaller than the high risk thresholds
# else: set low risk threshold = high risk threshold (eliminate the moderate risk group)
indexTh = 0
for th in lowThresholds:
    if(lowThresholds[indexTh] > highThresholds[indexTh]):
        lowThresholds[indexTh] = highThresholds[indexTh]
        print('WARNING for Day', indexTh, ': The high risk threshold is smaller than the low risk threshold. The low risk threshold is set to the high risk threshold. Therefore, the moderate risk group will be disbanded.')
    indexTh = indexTh + 1

# create dataframes
thr_dict = {'day': np.arange(31) , 'threshold': allThresholds, 'name': 'Threshold', 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1)}
thr_l_dict = {'day': np.arange(31) , 'threshold': lowThresholds, 'name': 'Low Risk', 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1)}
thr_h_dict = {'day': np.arange(31) , 'threshold': highThresholds, 'name': 'High Risk', 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1)}

dfThresholds = pd.DataFrame(data = thr_dict)
dfThresholdsL = pd.DataFrame(data = thr_l_dict)
dfThresholdsH = pd.DataFrame(data = thr_h_dict)


highLow_dict = {'day': np.arange(31), 'high': high, 'low': low,'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1)}
dfHighLow = pd.DataFrame(data = highLow_dict)
dfHighLow.to_csv(snakemake.output[13], sep = ',')

highLowQ_dict = {'day': np.arange(31), 'high': highQ, 'low': lowQ, 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1), 'highRiskName': 'High Risk', 'lowRiskName': 'Low Risk'}
dfHighLowQ = pd.DataFrame(data = highLowQ_dict)
dfHighLowQ.to_csv(snakemake.output[14], sep = ',')

highLowT_dict = {'day': np.arange(31), 'high': highThresholds, 'low': lowThresholds, 'x': np.arange(-0.5, 30.5, 1), 'x2': np.arange(0.5,31.5, 1), 'highRiskName': 'High Risk', 'lowRiskName': 'Low Risk'}
dfHighLowT = pd.DataFrame(data = highLowT_dict)
dfHighLowT.to_csv(snakemake.output[20], sep = ',')

# read in data containing the death dates
outcomes = pd.read_csv(snakemake.input[93], delimiter= ';').astype(str)

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


# plot pseudonyms that switch between binary outcomes
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


## plot pseudonyms, that do not switch between binary outcomes
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


## plot all pseudonyms
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


## plot all pseudonyms using outcomes as color
chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcome:O',sort = ['Alive', '0-100']),
    alt.Color("outcome:O",sort = ['Alive', '0-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518']))).properties(
    width=1200,
    height=1200)
chart.save(snakemake.output[6])

# plot all pseudonyms using three colours
chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    alt.Color("outcomeDay:O",sort = ['Alive','0-30', '31-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518','#54a24b']))).properties(
    width=1200,
    height=1200)
chart.save(snakemake.output[10])

# plot all pseudonyms using more colours
chart = alt.Chart(df).mark_point().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('detailedOutcomeDay:O',sort = ['Alive','76-100', '51-75', '31-50', '0-30']),
    alt.Color("detailedOutcomeDay:O",sort = ['Alive','0-30', '31-50', '51-75', '76-100'], scale = alt.Scale(range = ['#4c78a8','#e41a1c', '#ffbf79',  '#d67195', '#54a24b'])).title('Outcome')).properties(
    width=2200,
    height=1000)
chart.save(snakemake.output[19])

# plot all pseudonyms using three colours + threshold
dots = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q", scale=alt.Scale(domain=[0, 1])).title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    alt.Color("outcomeDay:O",sort = ['Alive','0-30', '31-100'], scale = alt.Scale(range = ['#4c78a8', '#f58518','#54a24b']))).properties(
    width=1200,
    height=1200)

line = alt.Chart(dfThresholds).mark_line().encode(
    x = "day:O",
    y = alt.Y("threshold:Q", scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2']))
)
chart = alt.layer(line, dots).resolve_scale(color = 'independent').configure_axisX(
titleFontSize=20, labelFontSize = 17, titlePadding = 15).configure_axisY(titleFontSize = 20, titlePadding = 20).configure_title(fontSize = 22).configure_legend(titleFontSize = 20, labelFontSize = 17)

chart.save(snakemake.output[11])


# Low High Mid
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

# Low High Mid Quantiles
dots = alt.Chart(df).mark_point().encode(
    alt.X("day:O", axis=alt.Axis(tickMinStep=1)).title('Day'),
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


highRisk = alt.Chart(dfHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "high:Q",
    color = alt.Color("highRiskName:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(dfHighLowQ).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "low:Q",
    color = alt.Color("lowRiskName:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(line,lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared").configure_axisX(tickMinStep=1.0)
chart.save(snakemake.output[15])

## Low High Mid Training Thresholds
dots = alt.Chart(df, title = plotTitle).mark_point().encode(
    alt.X("day:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])).title('Days'),
    alt.Y("probabilities:Q",scale=alt.Scale(domain=[0, 1])).title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '31-100', '0-30']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = ['#e41a1c', '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=2000,
    height=1000)

#line = alt.Chart(dfThresholds).mark_rule(strokeWidth=2).encode(
#    x ="x:O",
#    x2 = "x2:O",
#    y = "threshold:Q",
#    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))


highRisk = alt.Chart(dfThresholdsH).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O",axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("threshold:Q",scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N", scale = alt.Scale(range=['red'])).title('High Risk')
)

lowRisk = alt.Chart(dfThresholdsL).mark_rule(strokeWidth=2).encode(
    x =alt.X("x:O", axis = alt.Axis(values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])),
    x2 = "x2:O",
    y = alt.Y("threshold:Q",scale=alt.Scale(domain=[0, 1])),
    color = alt.Color("name:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared", x = "shared").configure_axisX(
titleFontSize=20, labelFontSize = 17, titlePadding = 15, values=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]).configure_axisY(titleFontSize = 20, labelFontSize = 17, titlePadding = 20).configure_title(fontSize = 22).configure_legend(titleFontSize = 20, labelFontSize = 17, symbolSize = 220, symbolStrokeWidth = 2)
chart.save(snakemake.output[18])
chart.save(snakemake.output[22])


# Low High Mid Quantiles Filtered Day 1 
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

filteredHighLowQ = dfHighLowQ[dfHighLowQ.day == 1]
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
chart.save(snakemake.output[16])

# Low High Mid Thresholds Filtered Day 1
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

filteredDfThresholdsH = dfThresholdsH[dfThresholdsH.day == 1]
highRisk = alt.Chart(filteredDfThresholdsH).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['red'])).title('High Risk')
)
filteredDfThresholdsL = dfThresholdsL[dfThresholdsL.day == 1]
lowRisk = alt.Chart(filteredDfThresholdsL).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['blue'])).title('Low Risk')
)


chart = alt.layer(lowRisk, highRisk, dots).resolve_scale(color = 'independent', y = "shared").configure_axisX(tickMinStep=1.0)
chart.save(snakemake.output[21])

## Day 28
filtered = df[df.day == 28]
dots = alt.Chart(filtered).mark_point().encode(
    alt.X("day:O", axis=alt.Axis(tickMinStep=1)).title('Day'),
    alt.Y("probabilities:Q").title('Probability'),
    alt.XOffset('outcomeDay:O',sort = ['Alive', '0-30', '31-100']),
    #alt.Shape('outcomeDay:O',sort = ['Alive', '0-30', '31-100'], scale = alt.Scale(range= ['circle', 'square', 'triangle'])).title('Outcome'),
    alt.Color('outcomeDay:O',sort = ['0-30','31-100', 'Alive'], scale = alt.Scale(range = [ '#ffbf79', '#4c78a8'])).title('Outcome')).properties(
    width=250,
    height=350)
filteredThresholds = dfThresholds[dfThresholds.day == 28]
line = alt.Chart(filteredThresholds).mark_rule(strokeWidth=2).encode(
    x ="x:O",
    x2 = "x2:O",
    y = "threshold:Q",
    color = alt.Color("name:N", scale = alt.Scale(range=['#b279a2'])).title('Threshold'))
filteredHighLowQ = dfHighLowQ[dfHighLowQ.day == 28]
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

#####################################

chart = alt.Chart(df).mark_circle().encode(
    alt.X("day:O").title('Day'),
    alt.Y("binary:O"),
    alt.Color("outcome:O",sort = ['Alive', '0-100'],scale = alt.Scale(range = ['#4c78a8', '#f58518']))).properties(
    width=1200,
    height=400)

chart.save(snakemake.output[7])


## plot all pseudonyms with bad outcomes
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

# plot all pseudonyms with good outcomes
filteredData = df[df.outcome == 'Alive']
chart = alt.Chart(filteredData).mark_line().encode(
    alt.X("day:O").title('Day'),
    alt.Y("probabilities:Q"),
    alt.Color("pseudonyms", legend = None)).properties(
    width=1200,
    height=1200)


chart.save(snakemake.output[24])

df.drop(['quantiles', 'lowHigh'], inplace= True, axis = 1)
df.to_csv(snakemake.output[23])
