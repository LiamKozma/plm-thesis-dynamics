python -c "
import pandas as pd
import glob

# Load and combine all results
files = glob.glob('tuning_result_*.csv')
if not files:
    print('No CSV files found!')
    exit()
    
df = pd.concat([pd.read_csv(f) for f in files])

print('\n🌟 FULL SWEEP RESULTS (Sorted by Sigma) 🌟')
print('='*75)
print(df.sort_values(by=['Base_Sigma', 'Oracle_Layers']).to_string(index=False))

print('\n🎯 SWEET SPOT MATCHES (Purity 50-70% | Promiscuity 40-60%) 🎯')
print('='*75)
sweet = df[(df['Purity_%'] >= 50) & (df['Purity_%'] <= 70) & 
           (df['Promiscuity_%'] >= 40) & (df['Promiscuity_%'] <= 60)]

if not sweet.empty:
    print(sweet.sort_values(by=['Promiscuity_%', 'Purity_%']).to_string(index=False))
else:
    print('No exact matches in the bullseye. Here are the 3 closest:')
    # Calculate distance to the ideal center (Purity 60, Promiscuity 50)
    df['Dist'] = abs(df['Purity_%'] - 60) + abs(df['Promiscuity_%'] - 50)
    print(df.sort_values('Dist').drop(columns=['Dist']).head(3).to_string(index=False))
"
