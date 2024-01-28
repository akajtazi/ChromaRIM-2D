from joblib import dump, load
import matplotlib.pyplot as plt
import numpy
import utils


if __name__ == '__main__':
    print('ChromaRIM v2 - Linear regression 2D based prediction')

    # Set input parameters
    c18_rmse = 1.5 * 36  # 1.5 x C18 RMSE we've got after creating linear model
    f5_rmse = 1.5 * 44  # 1.5 x F5 RMSE we've got after creating linear model
    c18_experimental_ri = 518  # C18 experimental retention index
    f5_experimental_ri = 462  # F5 experimental retention index
    actual_smile = "OC1=CC=CC=C1"  # SMILE
    file_name = "input/phenol-C6H6O.xlsx"  # Descriptors file
    smiles_column_name = "SMILES"  # SMIlES' column name within a descriptors file

    # Load data from Exel file
    descriptors_name, descriptors_list_t, smiles_list = utils.read_all_worksheets(file_name,
                                                                                  smiles_column=smiles_column_name)

    print("Number of unique structures : ", len(set(smiles_list)))

    # Since our 2D list is represented as (n_features, n_smiles),
    # we need to invert the matrix to (n_smiles, n_features),
    descriptors_list = numpy.array(descriptors_list_t).T.tolist()

    # Load feature names
    c18_feature_names = load("cache/c18_ri-feature-names-16.joblib")
    f5_feature_names = load("cache/f5_ri-feature-names-16.joblib")

    # Filter descriptors
    c18_filtered_descriptors_t = []
    for feature in c18_feature_names:
        index = descriptors_name.index(feature)
        column = descriptors_list_t[index]
        c18_filtered_descriptors_t.append(column)
    c18_filtered_descriptors = numpy.array(c18_filtered_descriptors_t).T.tolist()

    f5_filtered_descriptors_t = []
    for feature in f5_feature_names:
        index = descriptors_name.index(feature)
        column = descriptors_list_t[index]
        f5_filtered_descriptors_t.append(column)
    f5_filtered_descriptors = numpy.array(f5_filtered_descriptors_t).T.tolist()

    # Load our LinearRegression object that represents our trained model
    c18_linear_model = load("cache/c18_ri-linear-model.joblib")
    f5_linear_model = load("cache/f5_ri-linear-model.joblib")

    # Predict values based on the features the model needs and is trained to
    c18_predicted_values = c18_linear_model.predict(c18_filtered_descriptors)
    f5_predicted_values = f5_linear_model.predict(f5_filtered_descriptors)

    c18_accepted_smiles = set()
    for i, predicted_val in enumerate(c18_predicted_values):
        if predicted_val >= c18_experimental_ri - c18_rmse and predicted_val <= c18_experimental_ri + c18_rmse:
            c18_accepted_smiles.add(smiles_list[i])

    f5_accepted_smiles = set()
    for i, predicted_val in enumerate(f5_predicted_values):
        if predicted_val >= f5_experimental_ri - f5_rmse and predicted_val <= f5_experimental_ri + f5_rmse:
            f5_accepted_smiles.add(smiles_list[i])

    print("[c18] Number of SMILES: ", len(set(c18_accepted_smiles)))
    print(c18_accepted_smiles)

    print("[f5] Number of SMILES: ", len(set(f5_accepted_smiles)))
    print(f5_accepted_smiles)

    union_smiles = list(set(c18_accepted_smiles).union(set(f5_accepted_smiles)))
    print("[union] Number of SMILES: ", len(union_smiles))
    print(union_smiles)

    intersected_smiles = list(set(c18_accepted_smiles).intersection(set(f5_accepted_smiles)))
    print("[intersection] Number of SMILES: ", len(intersected_smiles))
    print(intersected_smiles)

    actual_smile_index = smiles_list.index(actual_smile)
    print("[c18] SMILES predicted RI value: ", c18_predicted_values[actual_smile_index])
    print("[f5]  SMILES predicted RI value: ", f5_predicted_values[actual_smile_index])

    # Plot accepted by C18 only (without actual)
    for smiles in c18_accepted_smiles - f5_accepted_smiles:
        index = smiles_list.index(smiles)
        x = c18_predicted_values[index]
        y = f5_predicted_values[index]
        plt.plot(x, y, 'x', color="royalblue")

    # Plot accepted by F5 only (without actual)
    for smiles in f5_accepted_smiles - c18_accepted_smiles:
        index = smiles_list.index(smiles)
        x = c18_predicted_values[index]
        y = f5_predicted_values[index]
        plt.plot(x, y, 'x', color="rebeccapurple")

    # Plot accepted by C18 and F5 (without actual)
    for smiles in set(intersected_smiles):
        index = smiles_list.index(smiles)
        x = c18_predicted_values[index]
        y = f5_predicted_values[index]
        plt.plot(x, y, 'x', color="deeppink")

    # Plot rejected by both C18 and F5 (without actual)
    for smiles in set(smiles_list) - c18_accepted_smiles - f5_accepted_smiles:
        index = smiles_list.index(smiles)
        x = c18_predicted_values[index]
        y = f5_predicted_values[index]
        plt.plot(x, y, 'x', color="lavender")

    # Plot actual SMILES in red
    # This needs to be plotted after all SMILES have been plotted, so it gets plotted above the already plotted one
    index = smiles_list.index(actual_smile)
    x = c18_predicted_values[index]
    y = f5_predicted_values[index]
    plt.plot(x, y, 'o', color="green")

    plt.xlabel("Predicted RI (C18)")
    plt.ylabel("Predicted RI (F5)")

    # Get min and max c18 and f5 values for plotting the lines
    min_x = min(c18_predicted_values) + 50
    max_x = max(c18_predicted_values) + 50
    min_y = min(f5_predicted_values) + 50
    max_y = max(f5_predicted_values) + 50

    plt.gca().set_xlim(min_x, max_x)
    plt.gca().set_ylim(min_y, max_y)

    # Plot c18 margin lines
    plt.plot([c18_experimental_ri - c18_rmse, c18_experimental_ri - c18_rmse], [min_y, max_y], linestyle='dashed', color='royalblue', linewidth=1, alpha=0.5)
    plt.plot([c18_experimental_ri + c18_rmse, c18_experimental_ri + c18_rmse], [min_y, max_y], linestyle='dashed', color='royalblue', linewidth=1, alpha=0.5)

    # Plot f5 margin lines
    plt.plot([min_x, max_x], [f5_experimental_ri - f5_rmse, f5_experimental_ri - f5_rmse], linestyle='dashed', color='rebeccapurple', linewidth=1, alpha=0.5)
    plt.plot([min_x, max_x], [f5_experimental_ri + f5_rmse, f5_experimental_ri + f5_rmse], linestyle='dashed', color='rebeccapurple', linewidth=1, alpha=0.5)

    # Save plot to output
    plt.savefig("output/prediction-f5-vs-c18.png", bbox_inches='tight', dpi=350)
    plt.clf()

    # Plot pie chart
    pie_total = len(set(smiles_list))
    pie_intersected = len(set(intersected_smiles))
    pie_c18_only = len(set(c18_accepted_smiles)) - pie_intersected
    pie_f5_only = len(set(f5_accepted_smiles)) - pie_intersected
    pie_rejected = pie_total - pie_c18_only - pie_f5_only - pie_intersected

    print("Rejected(%) = ", (1-pie_intersected/pie_total)*100, "%")

    y = numpy.array([pie_rejected, pie_c18_only, pie_intersected, pie_f5_only])
    colors = ["lavender", "royalblue", "deeppink", "rebeccapurple"]
    labels = ["Rejected", "Accepted C18", "Intersection", "Accepted F5"]
    explode = [0, 0, 0.1, 0]
    pie = plt.pie(y, colors=colors, explode=explode, autopct='%1.1f%%')
    plt.legend(pie[0], labels, loc="center right", bbox_to_anchor=(1, 0, 0.5, 1))

    # Save pie chart plot
    plt.savefig("output/prediction-pie-chart.png", bbox_inches='tight', dpi=350)
    plt.clf()
