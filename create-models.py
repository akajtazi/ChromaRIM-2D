from joblib import dump, load
import matplotlib.pyplot as plt
import numpy
import os
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
import utils


def feature_selection(ri_list, ri_name, descriptors_list, descriptors_name_list, number_of_features):
    # Calculate r-squared between every feature (X column) and retention index (y)
    # Exclude every feature with r-squared less than 0.1
    filtered_desc_list, filtered_desc_name_list = utils.regressionFeaturesFilter(ri_list, descriptors_list,
                                                                                 descriptors_name_list, 0.1)

    # Sequential feature selection (forward)
    sfs_features_file = "cache/" + ri_name + "-features-" + str(number_of_features) + ".joblib"
    sfs_feature_names_file = "cache/" + ri_name + "-feature-names-" + str(number_of_features) + ".joblib"
    ridge = RidgeCV().fit(filtered_desc_list, ri_list)

    try:  # Check if we have cached results
        sfs_features = load(sfs_features_file)
        sfs_feature_names = load(sfs_feature_names_file)
    except:
        print("Started sequential feature selector (", number_of_features, "). This might take a while.")
        sfs_forward = (
            SequentialFeatureSelector(ridge, n_features_to_select=number_of_features)
            .fit(filtered_desc_list, ri_list))
        sfs_features = sfs_forward.transform(filtered_desc_list)
        sfs_feature_names = sfs_forward.get_feature_names_out(filtered_desc_name_list)
        dump(sfs_features, sfs_features_file)
        dump(sfs_feature_names, sfs_feature_names_file)

    # Sort the features and plot them
    sfs_features_coefs = []
    for f in sfs_feature_names:
        index = filtered_desc_name_list.index(f)
        sfs_features_coefs.append(ridge.coef_[index])
    importance = numpy.abs(sfs_features_coefs)
    feature_names = numpy.array(sfs_feature_names)
    merged = list(zip(importance, feature_names))
    merged.sort(key=lambda elem: elem[0])
    split1, split2 = zip(*merged)
    sorted_indices = sorted(range(len(importance)), key=lambda k: importance[k], reverse=True)
    sorted_importance = [importance[i] for i in sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    bar_color = "royalblue" if ri_name == "c18_ri" else "rebeccapurple"
    plt.barh(width=sorted_importance, y=sorted_feature_names)
    plt.barh(width=split1, y=split2, color=bar_color)
    plt.title("Feature importances via coefficients")
    plt.xlim(0, 10)
    plt.savefig("output/features-" + ri_name + ".png", bbox_inches="tight", dpi=350)
    plt.clf()

    return sfs_features


def create_linear_model(ri_list, ri_name, descriptors_list, descriptors_name_list):
    number_of_features = 16  # Number of feature was acquired experimentally
    features = feature_selection(ri_list, ri_name, descriptors_list, descriptors_name_list, number_of_features)

    # Split dataset into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(features, ri_list, test_size=0.3, random_state=12)

    # Create linear regression model and store it in a .joblib file, so it can be used later for prediction
    reg = linear_model.LinearRegression()
    scores = cross_val_score(reg, X_train, y_train, scoring='r2', cv=5)
    reg.fit(X_train, y_train)
    dump(reg, "cache/" + ri_name + "-linear-model.joblib")

    # Now that we have a model, predict both train and test set
    y_test_predicted = reg.predict(X_test)
    y_train_predicted = reg.predict(X_train)

    # Plot predicted and actual values of train and test set
    color = "royalblue" if ri_name == "c18_ri" else "rebeccapurple"
    utils.do_plot(y_train, y_train_predicted, y_test, y_test_predicted, color=color, path="output/"+ri_name+".png")

    # Calculate and cache RMSE (train set)
    rmse_train = mean_squared_error(y_train, y_train_predicted, squared=False)
    dump(rmse_train, "cache/" + ri_name + "-linear-rmse.joblib")

    # Print some general information about the model
    print("##### ", ri_name, " LinearRegression model #####")
    print("Number of features: ", number_of_features)
    print("Cross validation score mean: ", scores.mean(), ", score dev: ", scores.std())
    print("r squared (train set): ", utils.calc_r_squared(y_train, y_train_predicted))
    print("MAE (train set):", mean_absolute_error(y_train, y_train_predicted))
    print("RMSE (train set):", mean_squared_error(y_train, y_train_predicted, squared=False))
    print("r squared (test set): ", utils.calc_r_squared(y_test, y_test_predicted))
    print("MAE (test set):", mean_absolute_error(y_test, y_test_predicted))
    print("RMSE (test set):", mean_squared_error(y_test, y_test_predicted, squared=False))
    print()


def create_random_forest_model(ri_list, ri_name, descriptors_list, descriptors_name_list):
    number_of_features = 16  # Number of feature was acquired experimentally
    features = feature_selection(ri_list, ri_name, descriptors_list, descriptors_name_list, number_of_features)

    X_train, X_test, y_train, y_test = train_test_split(features, ri_list, test_size=0.3, random_state=12)

    rfc = RandomForestRegressor(max_features=None)
    scores = cross_val_score(rfc, X_train, y_train, scoring='r2', cv=5)
    rfc.fit(X_train, y_train)
    dump(rfc, "cache/" + ri_name + "-random-forest-model.joblib")

    # Now that we have a model, predict both train and test set
    y_test_predicted = rfc.predict(X_test)
    y_train_predicted = rfc.predict(X_train)

    # Plot predicted and actual values of train and test set
    color = "royalblue" if ri_name == "c18_ri" else "rebeccapurple"
    utils.do_plot(y_train, y_train_predicted, y_test, y_test_predicted, color=color, path="output/"+ri_name+".png")

    # Calculate and cache RMSE (train set)
    rmse_train = mean_squared_error(y_train, y_train_predicted, squared=False)
    dump(rmse_train, "cache/" + ri_name + "-random-forest-rmse.joblib")

    # Print some general information about the model
    print("##### ", ri_name, " RandomForestRegressor model #####")
    print("Number of features: ", number_of_features)
    print("Cross validation score mean: ", scores.mean(), ", score dev: ", scores.std())
    print("r squared (train set): ", utils.calc_r_squared(y_train, y_train_predicted))
    print("MAE (train set):", mean_absolute_error(y_train, y_train_predicted))
    print("RMSE (train set):", mean_squared_error(y_train, y_train_predicted, squared=False))
    print("r squared (test set): ", utils.calc_r_squared(y_test, y_test_predicted))
    print("MAE (test set):", mean_absolute_error(y_test, y_test_predicted))
    print("RMSE (test set):", mean_squared_error(y_test, y_test_predicted, squared=False))
    print()


if __name__ == '__main__':
    print('ChromaRIM 2D - Create linear regression models')

    # Set input parameters (they depend on the descriptors input file)
    xlsx_filename = "input/descriptors.xlsx"
    main_worksheet_name = "main"
    main_ws = utils.read_worksheet(xlsx_filename, main_worksheet_name)
    c18_ri = utils.get_one_row_list_from_range(main_ws['E2':'E101'])  # C18 retention indices
    f5_ri = utils.get_one_row_list_from_range(main_ws['F2':'F101'])  # F5 retention indices
    smiles = utils.get_one_row_list_from_range(main_ws['G2':'G101'])  # SMILES

    # Load descriptors
    all_descriptors_name, all_descriptors_list_t = utils.read_all_worksheets(xlsx_filename, except_sheet=main_worksheet_name)

    # Since our 2D list is represented as (n_features, n_samples) and the scikit needs input as
    # (n_samples, n_features), we need to invert the matrix
    all_descriptors_list = numpy.array(all_descriptors_list_t).T.tolist()

    # Create cache folder if it doesn't exist for future purposes
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # Create output folder if it doesn't exist for future purposes
    if not os.path.exists("output"):
        os.makedirs("output")

    # With threshold = 0 we remove columns that have same values in each row
    selector = VarianceThreshold(threshold=0)
    new_descriptors_list = selector.fit_transform(all_descriptors_list)
    new_descriptors_name = selector.get_feature_names_out(all_descriptors_name)

    # Usually this is where normalization or standardization of descriptors would happen,
    # but this model gives better results without it

    # Create linear models, one using C18 and other using F5 retention indices
    create_linear_model(c18_ri, "c18_ri", new_descriptors_list, new_descriptors_name)
    create_linear_model(f5_ri, "f5_ri", new_descriptors_list, new_descriptors_name)

    # Create random forest models, one using C18 and other using F5 retention indices
    create_random_forest_model(c18_ri, "c18_ri", new_descriptors_list, new_descriptors_name)
    create_random_forest_model(f5_ri, "f5_ri", new_descriptors_list, new_descriptors_name)
