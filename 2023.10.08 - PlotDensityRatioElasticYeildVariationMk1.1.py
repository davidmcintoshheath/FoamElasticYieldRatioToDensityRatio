#!/usr/bin/python3

#David Heath of The University of Oxford. Created circa 2023.

#Dependacies
# polars
# matplotlib
# xlsx3csv


import matplotlib.pyplot
import numpy
import polars
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics

def main():
    plottingColours = ["darkorange","purple","yellowgreen","royalblue","slategrey","darkturquoise","gold"]
    #Import the data
    #Import the
    # fileName = str("2022.11.22 - MeshOptimisationMaterialModel - Pross 2023.02.23.xlsx")
    # fileName = str("2023.07.30 - MeshOptimisationMaterialModel - Pross .xlsx")
    #fileName = str("2023.08.05 - MeshOptimisationMaterialModel - Pross .xlsx")
    fileName = str("2023.10.06 - Material Characterisation at Different Densities.xlsx")
    #sheetName = str("VonMises Hardening BSI844")
    secondSheetName = str("PythonPlotELStressRatio")

    dataFrameTwo = polars.read_excel(fileName, sheet_name = secondSheetName)
    # print(dataFrameTwo)

    #Process the data
    #Generate experimental data vectors
    densityRatio = numpy.array(dataFrameTwo.get_column("Density Ratio (p* from ASTM F1839-08)").drop_nulls())
    modulusRatio = numpy.array(dataFrameTwo.get_column("Elastic yield plateu / solid modulus ratio").drop_nulls())
    averageModelCoefficient = numpy.array(dataFrameTwo.get_column("Average C4").drop_nulls())
    upper95ModelCoefficient = numpy.array(dataFrameTwo.get_column("Upper 95% CI").drop_nulls())
    lower95ModelCoefficient = numpy.array(dataFrameTwo.get_column("Lower 95% CI").drop_nulls())
    
    #Polynomial data to fit to the model
    #Density ratio valyues
    polynomialDensityRatioValues = numpy.linspace(min(densityRatio), max(densityRatio), 10)
    
    modulusCoefficientArray = numpy.linspace(lower95ModelCoefficient,upper95ModelCoefficient,100000)
    
    #Modulus values
    averagePolynomialModulusRatioValues = (polynomialDensityRatioValues ** 2) * averageModelCoefficient[0]
    upperPolynomialModulusRatioValues = (polynomialDensityRatioValues ** 2) * upper95ModelCoefficient[0]
    lowerPolynomialModulusRatioValues = (polynomialDensityRatioValues ** 2) * lower95ModelCoefficient[0]
    
    #Create polynomial parameters
    polynomialFeatureClassInstantiation = sklearn.preprocessing.PolynomialFeatures(degree = 2)
    polynomialFeatureParameters = polynomialFeatureClassInstantiation.fit_transform(polynomialDensityRatioValues.reshape(-1,1))
    
    #Select the type of regression
    linearFit = sklearn.linear_model.LinearRegression(fit_intercept = True, copy_X = True)
    
    currentBestR2 = 0
    
    currentModulusCoefficient = 0
    
    for modulusCoefficientIterator in range(len(modulusCoefficientArray)):
        
        fittingPolynomialModulusValues = (polynomialDensityRatioValues ** 2) * modulusCoefficientArray[modulusCoefficientIterator]
        
        currentModulusCoefficient = modulusCoefficientArray[modulusCoefficientIterator]
        
        #Run the fit
        linearFit.fit(polynomialFeatureParameters, fittingPolynomialModulusValues)
        # linearFit.fit(polynomialFeatureParameters, modulusRatio)
        
        linearCoefficient = linearFit.coef_
        # print(linearCoefficient)
        linearIntercept = float(linearFit.intercept_)
        
        fitPredictedModulusValues = linearFit.predict(polynomialFeatureClassInstantiation.transform(densityRatio.reshape(-1,1)))

        
        linearFitR2Score = sklearn.metrics.r2_score(modulusRatio, fitPredictedModulusValues)

        if currentBestR2 <= linearFitR2Score:
            currentBestR2 = linearFitR2Score
        else:
            break

    #Plot the data
    xAxis = [0,0]
    xAxisRange = [0, max(densityRatio)]
    
    #Create the plot
    fig, plot = matplotlib.pyplot.subplots()
    fitRoundSigFig = 4
    
    #Plot the raw data
    plot.scatter(densityRatio, modulusRatio, alpha = 0.80, linewidth = 0, color  = plottingColours[0], label = "Experimental Data")
    plot.plot(polynomialDensityRatioValues, averagePolynomialModulusRatioValues, color = plottingColours[3], label = "Average C4: " + str(numpy.round(averageModelCoefficient[0],fitRoundSigFig)))
    plot.fill_between(polynomialDensityRatioValues, upperPolynomialModulusRatioValues, lowerPolynomialModulusRatioValues, alpha = 0.25, linewidth = 0, color  = plottingColours[3], label = " C4 95% confidence interval")
    
    
    
    plot.plot(densityRatio, fitPredictedModulusValues, color = plottingColours[5], label = str("Best fit C4:" + " C4: " + str(numpy.round(currentModulusCoefficient[0],fitRoundSigFig)) + " R^2: " + str(numpy.round(currentBestR2,fitRoundSigFig))))

    #Plot the x axis
    plot.plot(xAxisRange, xAxis, linewidth = 0.75, linestyle = 'solid', color="black")
    
    #Create the legend
    plot.legend(loc = 'best', prop = {'size': 6})
    # plot.set_xscale('log')
    
    #Chagne the axes
    # matplotlib.pyplot.xlim([numpy.floor(max(densityRatio)), numpy.ceil(min(densityRatio))])
    matplotlib.pyplot.xlim([0.1, 0.4])
    matplotlib.pyplot.xlabel("Density Ratio")
    
    # matplotlib.pyplot.ylim([numpy.floor(min(modulusRatio)), numpy.ceil(max(modulusRatio))])
    matplotlib.pyplot.ylim([0.001, 0.015])
    # matplotlib.pyplot.ylim([numpy.floor(min(modulusRatio)), 0.3])
    matplotlib.pyplot.ylabel("Elastic Yield Ratio")
    
    #Normal
    fig.set_figwidth(6.4)
    fig.set_figheight(4.8)
    
    #Wide
    # fig.set_figwidth(9.6)
    # fig.set_figheight(5.9)

    #Show the plot
    matplotlib.pyplot.show()
    

if __name__ == "__main__":
    main()

