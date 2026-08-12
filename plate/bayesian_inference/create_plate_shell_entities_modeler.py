import KratosMultiphysics as Kratos


class CreatePlateShellEntitiesModeler(Kratos.Modeler):
    """Creates ShellThinElement3D3N per region SubModelPart and LineLoadCondition3D2N
    on the load edge. Each region gets its own Properties id so StructuralMaterials.json
    can assign a different YOUNG_MODULUS per zone."""

    def __init__(self, model: Kratos.Model, settings: Kratos.Parameters):
        super().__init__(model, settings)
        self.model = model
        self.settings = settings
        self.settings.AddMissingParameters(Kratos.Parameters("""{
            "model_part_name"        : "Structure",
            "source_sub_model_parts" : ["Region_1_alpha_100",
                                        "Region_2_alpha_50",
                                        "Region_3_alpha_080",
                                        "Region_4_alpha_100"],
            "load_sub_model_part"    : "Load_on_lines_Auto1",
            "element_name"           : "ShellThinElement3D3N",
            "condition_name"         : "LineLoadCondition3D2N"
        }"""))

    def SetupModelPart(self) -> None:
        super().SetupModelPart()

        root = self.model[self.settings["model_part_name"].GetString()]
        element_name = self.settings["element_name"].GetString()
        condition_name = self.settings["condition_name"].GetString()

        names = self.settings["source_sub_model_parts"].GetStringArray()
        for property_id, name in enumerate(names, start=1):
            properties = root.CreateNewProperties(property_id)
            source = root.GetSubModelPart(name)
            element_ids = []
            for geometry in source.Geometries:
                node_ids = [node.Id for node in geometry]
                root.CreateNewElement(element_name, geometry.Id, node_ids, properties)
                element_ids.append(geometry.Id)
            source.AddElements(sorted(element_ids))
            Kratos.Logger.PrintInfo("CreatePlateShellEntitiesModeler",
                                    "%s: %d elements, properties %d"
                                    % (name, len(element_ids), property_id))

        load_group = root.GetSubModelPart(self.settings["load_sub_model_part"].GetString())
        load_properties = root.CreateNewProperties(len(names) + 1)
        condition_ids = []
        for geometry in load_group.Geometries:
            node_ids = [node.Id for node in geometry]
            root.CreateNewCondition(condition_name, geometry.Id, node_ids, load_properties)
            condition_ids.append(geometry.Id)
        load_group.AddConditions(sorted(condition_ids))

        Kratos.Logger.PrintInfo("CreatePlateShellEntitiesModeler",
                                "%d conditions on %s"
                                % (len(condition_ids),
                                   self.settings["load_sub_model_part"].GetString()))


def Factory(model: Kratos.Model, settings: Kratos.Parameters) -> CreatePlateShellEntitiesModeler:
    return CreatePlateShellEntitiesModeler(model, settings)
