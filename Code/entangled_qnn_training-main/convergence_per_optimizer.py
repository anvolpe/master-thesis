import os
import json
import matplotlib.pyplot as plt
import numpy as np

# json dateien werden gefunden und richtig eingelesen :)
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            print(f"Lade Datei: {file_path}")
            with open(file_path, 'r') as file:
                try:
                    json_data = json.load(file)
                    data.append(json_data)
                except json.JSONDecodeError:
                    print(f"Fehler beim Laden der Datei: {file_path}")
    if not data:
        print("Keine JSON-Dateien gefunden oder alle Dateien sind fehlerhaft.")
    return data

def extract_optimizer_data(json_data):
    optimizers = ["nelder_mead", "bfgs", "cobyla", "sgd", "powell", "slsqp", "dual_annealing"] 
    optimizer_data = {}

    # cobyla wird aktuell gar nicht geplottet

    for entry in json_data:
        if isinstance(entry, dict):
            # alle databatches durchgehen
            for batch_key in entry:
                if batch_key.startswith("databatch_"):
                    print(f"Verarbeite Datenbatch: {batch_key}")
                    for optimizer in optimizers:
                        if optimizer in entry[batch_key]:
                            print(f"Verarbeite Optimierer: {optimizer}")

                            # liste für optimierer
                            if optimizer not in optimizer_data:
                                optimizer_data[optimizer] = []

                            # daten für jeden durchlauf entnehmen
                            batch_data = entry[batch_key][optimizer]
                            for key in batch_data:
                                data = batch_data[key]
                                #print(f"Durchlauf-Daten: {data}")
                                
                                # data muss dictionary sein und schlüssel enthalten
                                if isinstance(data, dict):
                                    nit = data.get("nit", None)
                                    fun = data.get("fun", None)
                                    
                                    if nit is not None and fun is not None:
                                        try:
                                            nit = int(nit)
                                            fun = float(fun)
                                            optimizer_data[optimizer].append((nit, fun))
                                        except ValueError as e:
                                            print(f"Fehler beim Konvertieren der Daten: {e}")
                                    else:
                                        # tritt öfter auf: fehlt nit an manchen Stellen in den .json-Dateien ???
                                        print(f"Fehlende Schlüssel in den Daten: {data}")
                                else:
                                    # gibt hier aktuell öfter gradient/gradient-free aus
                                    print(f"Unerwartete Datenstruktur: {data}")
                        else:
                            print(f"Optimierer {optimizer} nicht in den Datenbatch {batch_key} gefunden")
        else:
            print("Eintrag ist kein Dictionary")

    #print("Extrahierte Optimierungsdaten:")
    #for optimizer, results in optimizer_data.items():
        #print(f"{optimizer}: {results}")

    # mean value berechnen für die Plots
    ### Standard deviation bisher nicht eingebaut ###
    mean_optimizer_data = {}
    for optimizer, results in optimizer_data.items():
        if results:
            print(f"Berechne Mittelwerte für: {optimizer}")
            results.sort()  # nach nits sortieren
            nits, funs = zip(*results)  # entpacken in nits und funs
            unique_nits = sorted(set(nits))
            mean_funs = [np.mean([fun for nit, fun in results if nit == unique_nit]) for unique_nit in unique_nits]
            mean_optimizer_data[optimizer] = (unique_nits, mean_funs)
    
    return mean_optimizer_data

def plot_optimizer_data(optimizer_data, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for optimizer, (nit, fun) in optimizer_data.items():
        if len(nit) > 0 and len(fun) > 0:
            plt.figure()
            plt.plot(nit, fun, marker='o', label=optimizer)
            plt.xlabel('Number of Iterations (nit)')
            plt.ylabel('Average Function Value (fun)')
            plt.title(f'Convergence Plot for {optimizer}')
            plt.legend()
            plt.xscale('log') # hier evtl abändern, damit man klarer sehen kann welcher nit vorkommen
            plt.grid(True)
            file_path = os.path.join(save_path, f'{optimizer}_convergence_plot.png')
            plt.savefig(file_path)
            # print(f"Plot gespeichert: {file_path}")
            plt.close()
        else:
            print(f"Keine Daten zum Plotten für Optimierer: {optimizer}")

def main(json_directory, save_path):
    json_data = load_json_files(json_directory)
    if not json_data:
        print("Keine Daten zum Verarbeiten.")
        return
    optimizer_data = extract_optimizer_data(json_data)
    if not optimizer_data:
        print("Keine Optimierungsdaten zum Plotten.")
        return
    plot_optimizer_data(optimizer_data, save_path)

if __name__ == "__main__":
    json_directory = 'qnn-experiments/experimental_results/results/2024-07-19_allConfigs_allOpt'
    save_path = 'qnn-experiments/experimental_results/results/optimizer_plots'
    main(json_directory, save_path)
