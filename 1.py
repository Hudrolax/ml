from pprint import pprint
import pandas as pd

with open('/Users/hudro/google_drive/obsidian/Base/страны.md') as f:
    df = pd.DataFrame(f.readlines(), columns=['Страны'])

df['Страны'] = df['Страны'].str.replace('\n', '')
pprint(df)
