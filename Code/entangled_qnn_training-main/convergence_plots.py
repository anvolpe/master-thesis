import os
import json
import matplotlib.pyplot as plt
import numpy as np

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
    gradient_based = ["nelder_mead", "powell", "cobyla"]
    gradient_free = ["sgd", "adam", "rmsprop", "bfgs", "dual_annealing", "slsqp"]
    optimizers = gradient_based + gradient_free

    optimizer_data = {}
    gradient_based_data = []
    gradient_free_data = []

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
                                
                                # data muss dictionary sein und schlüssel enthalten
                                if isinstance(data, dict):
                                    nit = data.get("nit", None)
                                    fun = data.get("fun", None)
                                    
                                    if nit is not None and fun is not None:
                                        try:
                                            nit = int(nit)
                                            fun = float(fun)
                                            optimizer_data[optimizer].append((nit, fun))
                                            if optimizer in gradient_based:
                                                gradient_based_data.append((nit, fun))
                                            if optimizer in gradient_free:
                                                gradient_free_data.append((nit, fun))
                                        except ValueError as e:
                                            print(f"Fehler beim Konvertieren der Daten: {e}")
                                    else:
                                        print(f"Fehlende Schlüssel in den Daten: {data}")
                                else:
                                    print(f"Unerwartete Datenstruktur: {data}")
                        else:
                            print(f"Optimierer {optimizer} nicht in den Datenbatch {batch_key} gefunden")
        else:
            print("Eintrag ist kein Dictionary")

    # Berechne mean fun values
    def calculate_mean_data(data):
        if not data:
            return [], []
        data.sort()
        nits, funs = zip(*data)
        unique_nits = sorted(set(nits))
        mean_funs = [np.mean([fun for nit, fun in data if nit == unique_nit]) for unique_nit in unique_nits]
        return unique_nits, mean_funs

    mean_optimizer_data = {optimizer: calculate_mean_data(results) for optimizer, results in optimizer_data.items()}
    mean_gradient_based_data = calculate_mean_data(gradient_based_data)
    mean_gradient_free_data = calculate_mean_data(gradient_free_data)
    
    return mean_optimizer_data, mean_gradient_based_data, mean_gradient_free_data, optimizer_data

def plot_optimizer_data(optimizer_data, optimizer_save_path):
    if not os.path.exists(optimizer_save_path):
        os.makedirs(optimizer_save_path)
        
    for optimizer, (nit, fun) in optimizer_data.items():
        if len(nit) > 0 and len(fun) > 0:
            plt.figure()
            plt.plot(nit, fun, marker='o', label=optimizer)
            plt.xlabel('Number of Iterations (nit)')
            plt.ylabel('Average Function Value (fun)')
            plt.title(f'Convergence Plot for {optimizer}')
            plt.legend()
            plt.xscale('log')
            plt.ylim(bottom=-0.015,  top=max(fun) + 0.05)
            plt.grid(True)
            file_path = os.path.join(optimizer_save_path, f'{optimizer}_convergence_plot.png')
            plt.savefig(file_path)
            plt.close()
        else:
            print(f"Keine Daten zum Plotten für Optimierer: {optimizer}")

def plot_category_data(nit, fun, category_name, category_save_path):
    if len(nit) > 0 and len(fun) > 0:
        plt.figure()
        plt.plot(nit, fun, marker='o', label=category_name)
        plt.xlabel('Number of Iterations (nit)')
        plt.ylabel('Average Function Value (fun)')
        plt.title(f'Convergence Plot for {category_name}')
        plt.legend()
        plt.xscale('log')
        plt.ylim(bottom=-0.02) 
        plt.grid(True)
        file_path = os.path.join(category_save_path, f'{category_name}_convergence_plot.png')
        plt.savefig(file_path)
        plt.close()
    else:
        print(f"Keine Daten zum Plotten für Kategorie: {category_name}")

def plot_boxplots(optimizer_data, boxplot_save_path): # sgd scheint viele ausreißer zu haben
    if not os.path.exists(boxplot_save_path):
        os.makedirs(boxplot_save_path)
        
    plt.figure()
    data_to_plot = []
    labels = []
    
    for optimizer, results in optimizer_data.items():
        if results:
            labels.append(optimizer)
            _, fun = zip(*results)
            data_to_plot.append(fun)
    
    plt.boxplot(data_to_plot, labels=labels)
    plt.xlabel('Optimizers')
    plt.ylabel('Function Value (fun)')
    plt.title('Boxplot of Function Values')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    file_path = os.path.join(boxplot_save_path, 'optimizer_boxplots.png')
    plt.savefig(file_path)
    plt.close()

def main(json_directory, optimizer_save_path, category_save_path, boxplot_save_path):
    json_data = load_json_files(json_directory)
    if not json_data:
        print("Keine Daten zum Verarbeiten.")
        return
    optimizer_data, gradient_based_data, gradient_free_data, raw_optimizer_data = extract_optimizer_data(json_data)
    if not optimizer_data:
        print("Keine Optimierungsdaten zum Plotten.")
        return
    plot_optimizer_data(optimizer_data, optimizer_save_path)
    plot_category_data(*gradient_based_data, 'Gradient-Based', category_save_path)
    plot_category_data(*gradient_free_data, 'Gradient-Free', category_save_path)
    plot_boxplots(raw_optimizer_data, boxplot_save_path)

if __name__ == "__main__":
    json_directory = 'qnn-experiments/experimental_results/results/2024-07-19_allConfigs_allOpt'
    optimizer_save_path = 'qnn-experiments/experimental_results/results/optimizer_plots'
    category_save_path = 'qnn-experiments/experimental_results/results/category_plots'
    boxplot_save_path = 'qnn-experiments/experimental_results/results/box_plots'
    main(json_directory, optimizer_save_path, category_save_path, boxplot_save_path)
