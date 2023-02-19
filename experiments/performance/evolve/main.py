import pandas as pd


df = pd.read_csv('evolve.tsv', sep='\t')
df.loc[df['version'].str.startswith('bias'), 'version'] = 'v4'

df = df.groupby(['version', 'n']).agg({'elapsed': 'median'}).reset_index()
df = df.rename(columns={
    'elapsed': 'Median Time Elapsed',
    'version': 'Version',
})
df = df.pivot(index='Version', columns='n')
df = df.round(1)
df = df.astype('str')
df = df + 's'

print(df.to_latex())
