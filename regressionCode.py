import numpy as np
import plotly.graph_objects as go
import re
from sympy import symbols, sympify


def parse_input(input_string):
    # Add a '*' between a digit and 'x' if not already present
    input_string = re.sub(r'(\d)([xX])', r'\1*\2', input_string)
    
    # Add a '1*' before 'x' if coefficient is not present
    input_string = re.sub(r'(?<!\d)([xX])', r'1*\1', input_string)

    # Replace '^' with '**' for exponentiation
    input_string = input_string.replace('^', '**')

    return input_string

def convert_text(text):
    text = str(text)
    normal_chars = "0123456789"
    superscript_chars = "⁰¹²³⁴⁵⁶⁷⁸⁹"

    mapping = str.maketrans(normal_chars, superscript_chars)

    converted_text = text.translate(mapping)
    return converted_text

def create_dist(x_mean: float, x_std: float, y_equation: str, y_noise_mean: float, y_noise_std, num_of_points: int, order: int, x_dist: str, y_dist: str):
    # Generate your x and y data points
    if(x_dist == "normal"):
        x = np.random.normal(x_mean, x_std, num_of_points)
    else:
        xlow = x_mean-(x_std/2)
        xhigh = x_mean+(x_std/2)
        x = np.random.uniform(xlow, xhigh, num_of_points)
    
    # Create a sympy symbol for 'x'
    x_sym = symbols('x')

    # Parse the equation string into a sympy expression
    y_equation_parsed = sympify(parse_input(y_equation))

    # Evaluate the expression for the generated 'x' values
    y = np.array([float(y_equation_parsed.subs(x_sym, val)) for val in x])
    
    # Add noise to 'y'
    if(y_dist=="normal"):
        y += np.random.normal(y_noise_mean, y_noise_std, num_of_points)
    else:
        ylow = y_noise_mean-(y_noise_std/2)
        yhigh = y_noise_mean+(y_noise_std/2)
        y += np.random.uniform(ylow, yhigh, num_of_points)

    # Add a column of ones to account for the y intercept constant
    A = np.column_stack((np.ones(x.shape[0]), x))
    for i in range(2, order + 1):
        A = np.column_stack((A, x**i))


    """START OF CALCULATIONS"""

    # Generating x hat for Ax = b
    def generate_x_hat(A: np.array):
        x_hat = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(y)
        return x_hat

    x_hat = generate_x_hat(A)

    # Calculate error 
    # y - p where p = A(x_hat)
    p = A.dot(x_hat)
    error = y - p

    # Calculate R^2 -- SSR/SST 
    # SSR = sum squared regression (SSR) = sum of errors squared
    # SST = total sum of squares (SST) = (sum of y values of points) - (mean of all y values)
    ## mean of y points
    
    ssr = np.sum(error ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    #if (sst > -.001 and sst < .001):
    if(sst == 0):
        r_squared = 0
    else:
        r_squared = 1-(ssr/sst)
    #print(f"ssr: {ssr}")
    #print(f"sst: {sst}")
    #print(f"R^2 ssr/sst: {r_squared}")
    

    """END OF CALCULATIONS"""


    """CREATING PLOTS"""
    # Plotting with Plotly
    fig = go.Figure()

    # Generate x values for the curve
    x_curve = np.linspace(np.min(x), np.max(x), 100)
    y_curve = np.zeros_like(x_curve)  # Initialize y_curve with zeros

    # Add terms for each power of x_curve
    for i in range(order + 1):
        y_curve += x_hat[i] * (x_curve ** i)  
    
    # Create the string equation 
    equation = "y = "
    for i in range(order, -1, -1):
        if i == 0:
            equation += f"{x_hat[i]:.2f}"
        elif i == 1:
            equation += f"{x_hat[i]:.2f}x + "
        else:
            equation += f"{x_hat[i]:.2f}x{convert_text(i)} + "
        if i%3 == 0:
            equation += "<br>"

    # Used to edit the y equation output to the legend of graph
    def add_br_every_fourth_space(input_string):
        words = input_string.split()
        updated_string = ""
        count = 0

        for word in words:
            updated_string += word + " "
            count += 1
            if count % 4 == 0:
                updated_string += "<br>"

        return updated_string.strip()

    # Scatter plot
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))

    # Line of best fit
    fig.add_trace(go.Scatter(x=x_curve, y = y_curve, mode='lines', name=f"{equation}", marker=dict(size=7, color="orangered", symbol='square')))

    #Add R^2
    fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"R²: {r_squared}<br>",
                    marker=dict(size=7, color="orangered", symbol='square'),
                ))
    
    # Add entered data
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"# of data points: {num_of_points}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))

    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"x_data_mean: {x_mean}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"x_dist: {x_dist}",
            marker=dict(size=7, color="orange", symbol='square'),
        ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"x_data_std/range: {x_std}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"y_data_equation: {add_br_every_fourth_space(y_equation)}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"y_noise_mean: {y_noise_mean}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"y_noise_dist: {y_dist}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"y_noise_std/range: {y_noise_std}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))
    
    fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"order of model: {order}",
                marker=dict(size=7, color="orange", symbol='square'),
            ))

    # Update layout for better visualization
    fig.update_layout(
        title="Interactive Scatter Plot with Line of Best Fit",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        height=600, 
        width=1080
    )

    return fig