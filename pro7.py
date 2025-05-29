import pyforest

def linear_regression_california():
    X, y = fetch_california_housing(as_frame=True).data[["AveRooms"]], fetch_california_housing().target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test); plt.plot(X_test, y_pred, color='red'); plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cyl", "disp", "hp", "wt", "acc", "yr", "org"]
    df = pd.read_csv(url, sep=r'\s+', names=cols, na_values="?").dropna()
    X, y = df[["disp"]], df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression()).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test); plt.scatter(X_test, y_pred, color='red'); plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

linear_regression_california()
polynomial_regression_auto_mpg()
