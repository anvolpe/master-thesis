import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV einlesen
df = pd.read_csv("qnn-experiments-main\\optimization_results.csv")

# Listen, die aus CSV später rausgelesen werden
optimizers = []
fun_values = []
nfev_values = []
durations = []
nits = []
successes = []

# Muster für Auslesen von Infos
fun_pattern = re.compile(r'fun:\s*([\d\.]+)')
nfev_pattern = re.compile(r'nfev:\s*(\d+)')
nit_pattern = re.compile(r'nit:\s*(\d+)')
success_pattern = re.compile(r'success:\s*(True|False)')

# Alle Zeilen des Dataframes durchiterieren
for _, row in df.iterrows():
    optimizer = row['Optimizer']
    result_text = row['Result']
    duration = row['Duration']
    
    fun_match = fun_pattern.search(result_text)
    fun_value = float(fun_match.group(1)) if fun_match else None
    
    nfev_match = nfev_pattern.search(result_text)
    nfev_value = int(nfev_match.group(1)) if nfev_match else None

    nit_match = nit_pattern.search(result_text)
    nit_value = int(nit_match.group(1)) if nit_match else None

    success_match = success_pattern.search(result_text)
    success_value = success_match.group(1) == 'True' if success_match else None
    
    # extrahierten Daten zu Listen einfügem
    optimizers.append(optimizer)
    fun_values.append(fun_value)
    nfev_values.append(nfev_value)
    durations.append(duration)
    nits.append(nit_value)
    successes.append(success_value)

results_df = pd.DataFrame({
    'Optimizer': optimizers,
    'Fun Value': fun_values,
    'NFEV': nfev_values,
    'Duration': durations,
    'Nit': nits,
    'Success': successes
})

print(results_df.head())

plt.figure(figsize=(12, 6))

# Plot für Funktionswerte
plt.subplot(1, 5, 1)
plt.bar(results_df['Optimizer'], results_df['Fun Value'])
plt.xlabel('Optimizer')
plt.ylabel('Fun Value')
plt.title('Final Fun Value by Optimizer')
plt.xticks(rotation=45, ha='right')

# Plot für Anzahl der Funktionsaufrufe
plt.subplot(1, 5, 2)
plt.bar(results_df['Optimizer'], results_df['NFEV'])
plt.xlabel('Optimizer')
plt.ylabel('NFEV')
plt.title('Number of Function Evaluations')
plt.xticks(rotation=45, ha='right')

#Plot für Duration
plt.subplot(1,5,3)
plt.bar(results_df['Optimizer'], results_df['Duration'])
plt.xlabel('Optimizer')
plt.ylabel('Duration')
plt.title('Duration of Optimizer')
plt.xticks(rotation=45, ha='right')

# Plot für Number of Iterations
# Falls es kein Nit gibt, also NaN, dann wird aktuell der Optimizer gar nicht dargestellt
plt.subplot(1,5,4)
plt.bar(results_df['Optimizer'], results_df['Nit'])
plt.xlabel('Optimizer')
plt.ylabel('Nit')
plt.title('Number of Iterations')
plt.xticks(rotation=45, ha='right')

# Success 
plt.subplot(1,5,5)
plt.bar(results_df['Optimizer'], results_df['Success'])
plt.xlabel('Optimizer')
plt.ylabel('Success')
plt.title('Success of Optimizer')
plt.xticks(rotation=45, ha='right')
plt.yticks([0, 1], ['False', 'True'])


plt.tight_layout()
plt.show()
