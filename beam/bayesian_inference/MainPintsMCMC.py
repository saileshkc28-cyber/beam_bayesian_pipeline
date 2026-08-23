import KratosMultiphysics as Kratos

from mcmc_analysis import MCMCAnalysis


if __name__ == "__main__":
    model = Kratos.Model()

    with open("PintsMCMCParameters.json", "r") as file_input:
        parameters = Kratos.Parameters(file_input.read())

    analysis = MCMCAnalysis(model, parameters)
    analysis.Run()
