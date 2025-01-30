import pandas as pd
import numpy as np
import altair as alt
import vl_convert as vlc
import re

numberOfDays = snakemake.config['stop'];

# could do this with an array and find function, works now
# !! the order in which the parameters are checked is very important !!
# e.g. if K is checked after FK506, FK506 will not be in the resulting data anymore
# it will be misinterpreted as K
# function used to create the data for the compressed FI heatmap
def shortedNames(row):

    name = row['Names']

    if("MCH" in row['Names']):
        name = "MCH"
    if("THROMB" in row['Names']):
        name = "THROMB"
    if("MCV" in row['Names']):
        name = "MCV"
    if("EVB" in row['Names']):
        name = "EVB"
    if("ERY" in row['Names']):
        name = "ERY"
    if("MCHC" in row['Names']):
        name = "MCHC"
    if("K" in row['Names']):
        name = "K"
    if("LEUKO" in row['Names']):
        name = "LEUKO"
    if("HB" in row['Names']):
        name = "HB"
    if("HK" in row['Names']):
        name = "HK"
    if("CRP" in row['Names']):
        name = "CRP"
    if("HSRE" in row['Names']):
        name = "HSRE"
    if("BILI" in row['Names']):
        name = "BILI"
    if("CA" in row['Names']):
        name = "CA"
    if("CREA" in row['Names']):
        name = "CREA"
    if("GLUC-S" in row['Names']):
        name = "GLUC-S"
    if("HST" in row['Names']):
        name = "HST"
    if("GFR-MDRD" in row['Names']):
        name = "GFR-MDRD"
    if("GFR-CKD" in row['Names']):
        name = "GFR-CKD"
    if("NRBC-ABS" in row['Names']):
        name = "NRBC-ABS"
    if("FK506" in row['Names']):
        name = "FK506"
    if("IL-6" in row['Names']):
        name = "IL-6"

    if("BFA" in row['Names']):
        name = 'Resistenz'
    if("BFN" in row['Names']):
        name = 'Resistenz'

    return name

### HYPERPARAMETERS
j = 0;

# determine model that was used
model = ''
if(snakemake.wildcards.model == 'random_forest'):
    model = 'Random Forest'
elif(snakemake.wildcards.model == 'gradient_boosting'):
    model = 'Gradient Boosting'

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

# read in hyperparameters data
complete = []
for i in numberOfDays:

    data = pd.read_csv(snakemake.input[j], delimiter= ',').astype(str)
    data.drop(columns = ['Unnamed: 0'], inplace = True)
    complete.append(data)

    j = j + 1;

if(model == 'Random Forest'):
    df = pd.concat(complete)
    df.rename(columns={'model__max_depth':'maxDepth', 'model__max_features':'maxFeatures', 'model__min_samples_leaf':'minSamplesLeaf', 'model__n_estimators' :'nEstimators', 'selector__k':'k'}, inplace = True)
    df.reset_index(inplace=True)
    df.drop(columns = ['index'], inplace = True)
    df.reset_index(inplace=True)

    altDataRF = pd.melt(df,id_vars=['index'], value_vars=['maxDepth', 'maxFeatures', 'minSamplesLeaf', 'nEstimators', 'k'])

elif(model == 'Gradient Boosting'):
    df = pd.concat(complete)
    df.rename(columns={'model__learning_rate': 'learningRate', 'model__subsample':'subsample','model__max_depth':'maxDepth', 'model__max_features':'maxFeatures', 'model__min_samples_leaf':'minSamplesLeaf', 'model__n_estimators' :'nEstimators', 'selector__k':'k'}, inplace = True)
    df.reset_index(inplace=True)
    df.drop(columns = ['index'], inplace = True)
    df.reset_index(inplace=True)

    altDataRF = pd.melt(df,id_vars=['index'], value_vars=['learningRate', 'maxDepth', 'maxFeatures', 'minSamplesLeaf', 'nEstimators', 'subsample', 'k'])



# save hyperparameters table
altDataRF.to_csv(snakemake.output[2])

# plot hyperparameters
base = alt.Chart(altDataRF, title = plotTitle).transform_joinaggregate(
    test='count(value)',
    groupby=['variable']
).encode(
    alt.X("index:O").title("Days").axis(),
    alt.Y("variable:N").title("Hyperparameters")).properties(
    width=1500,
    height=250
)
rect = base.mark_rect().encode(
    alt.Color("value", legend = None#, scale = alt.Scale(range = ['#4c78a8', '#f58518', '#e45756', '#72b7b2', '#54a24b', '#eeca3b', '#b279a2', '#ff9da6', '#9d755d', '#bab0ac'])
    ).title(None),
)

text = base.mark_text(baseline='middle', fontSize = 14).encode(
    alt.Text('value')
)
hyperparametersPlot = (rect+text).configure_range(
    category={'scheme': 'tableau20'}
).configure_axisX(titleFontSize=16, labelFontSize = 14, titlePadding = 10).configure_axisY(titleFontSize = 16, labelFontSize = 14, titlePadding = 20).configure_title(fontSize = 18)
hyperparametersPlot.save(snakemake.output[0])
hyperparametersPlot.save(snakemake.output[1])

hyperparametersPlot = (rect+text)

### FEATURE IMPORTANCE

# read in feature importance data
i = 0
j = 31;
completeDT = []
completeRF = []
completeGB = []
columnsDT = []
columnsRF = []
columnsGB = []

for i in numberOfDays:
    if(j < 0):
        print('not yet')
    else:
        rf = pd.read_csv(snakemake.input[j], delimiter = ',');
        # only use top 10 features (are already sorted)
        #rf = rf.loc[0:9]
        rf.set_index('Names', inplace= True)
        rf.drop(columns=['Unnamed: 0'], inplace=True)
        rf = rf.T
        rf.loc[:, 'Day'] = int(i)
        rf.set_index('Day', inplace=True)
        # drop the day from the names of the time series parameters
        renameDict = {}
        for c in rf.columns.values:
            if '__w=' in c:
                renameDict[c] = re.sub('__w=\d{1,2}', '', c)
        rf.rename(columns=renameDict, inplace=True)
        columnsRF.extend(rf.columns.values)
        completeRF.append(rf)
    i = i + 1;
    j = j + 1;


dfRF = pd.concat(completeRF)
dfRF.reset_index(inplace=True)
altDataRF = pd.melt(dfRF,id_vars=['Day'], value_vars=columnsRF)


# save FI table
altDataRF.to_csv(snakemake.output[3])


#### Paper Figures ####

## Figure 4 ##

fi = altDataRF

def group(value):
    split = value.split()
    if split[0][1] == '-': #Ops Codes
        return 'OPS-Code'
    if '__' in split[0]: #TS
        base_value = split[0].split('_')[0]
        if(base_value in ['CRP', 'IL-6']):
            return 'Inflammation'
        else:
            return base_value
    if split[0][0].isalpha() == True and split[0][0].isupper() and split[0][1].isnumeric(): # ICD-Code
        return 'ICD-Code'
    if len(split[0].split('_')) == 2: # Measurement
        base_value = split[0].split('_')[0]
        if(base_value in ['CRP', 'IL-6']):
            return 'Inflammation'
        else:
            return base_value
    if split[0] == 'Alter': # Alter
        return 'Age'
    if split[0] == 'Geschlecht': #Geschlecht
        return 'Sex'
    if value.startswith('BFA') or value.startswith('BFN') or value in ['MRGN','VRE','MRSA']:
        return 'Resistance/Pathogens'
    if split[0][0].islower():
        return 'Comorbidities'
    return value

fi['Group'] = fi['Names'].apply(group)

statics = [
        'Age',
        'Sex',
        'Comorbidities',
        'Resistance/Pathogens',
        'ICD-Code',
        'OPS-Code'
]

rest = fi.groupby('Group',as_index=False)['value'].sum().sort_values(by='value')['Group'][::-1]

rest = [r for r in rest if r not in statics]

combined = statics + rest


gb = fi.groupby(['Day','Group'],as_index=False)['value'].sum()
gb['zero'] = gb['value'] == 0

condition=alt.condition(
    alt.datum.zero,
    alt.value("lightgrey"),
    alt.Color('value',title='Feature Importance',scale=alt.Scale(domain=[0.6,0],scheme='inferno'))
)

fig = alt.Chart(gb).mark_rect().encode(
    x='Day:O',
    color=condition,
    y=alt.Y('Group',title='Feature (Group)',sort=combined),
    tooltip='value'
)

fig.save(snakemake.output[4])



## Supplementary Figure 2 ##

fi = altDataRF

def group(value):
    split = value.split()
    if len(split[0].split('_')) == 2: # Measurement
        base_value = split[0].split('_')[0]
        return 'Measurement'
    return 'Other'



fi['Group'] = fi['Names'].apply(group)

fi = fi[fi['Group']=='Measurement'][['Names','value','Day']]
fi['Measurement Day'] = fi['Names'].apply(lambda x : int(x.split('_')[-1]))
fi = fi.groupby(['Day','Measurement Day'],as_index=False)['value'].sum()

fi['zero'] = fi['value'] == 0

fi['nonexistant'] = fi['Measurement Day'] > fi['Day']

condition=alt.condition(
    alt.datum.nonexistant,
    alt.value("white"),
    alt.Color('value',title='Feature Importance',scale=alt.Scale(domain=[0.3,0],scheme='inferno'))
)

condition2 =     alt.condition(
     alt.datum.zero,
     alt.value('lightgrey'),
     alt.Color('value',title='Feature Importance',scale=alt.Scale(domain=[0.3,0],scheme='inferno'))
)

fig = alt.Chart(fi).mark_rect().encode(
    x=alt.X('Measurement Day:O'),
    y = alt.Y('Day:O',title='Prediction Day'),
    color=condition,
    tooltip=['value']
)+alt.Chart(fi[~fi['nonexistant']]).mark_rect().encode(
    x=alt.X('Measurement Day:O'),
    y = alt.Y('Day:O',title='Prediction Day'),
    color=condition2,
    tooltip=['value']
)
fig.save(snakemake.output[5])
