import streamlit as st
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error, r2_score
import dask.array as da
import matplotlib.pyplot as plt

df = dd.read_csv("advertising.csv")

df.head()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

columns = ['TV', 'Radio', 'Newspaper']

for i, col in enumerate(columns):
    axes[i].scatter(df[col].compute(), df['Sales'].compute())
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Sales')
    axes[i].set_title(f'{col} vs. Sales')

# Adjust the spacing between subplots
plt.tight_layout()

# Display the scatter plots
st.pyplot(fig)
X_dask = df[['TV', 'Radio', 'Newspaper']].to_dask_array(lengths=True)
y_dask = df['Sales'].to_dask_array(lengths=True)

X_train, X_test, y_train, y_test = train_test_split(X_dask, y_dask, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
# Create a scatter plot of the actual vs. predicted values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs. Predicted Sales')

# Display the scatter plot
st.pyplot(fig)

# Display the evaluation results
st.title("Linear Regression Model Results")
st.write("Mean Squared Error:", mse)
st.write("R-squared:", r2)
