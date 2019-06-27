from collections import OrderedDict

COMPONENTS = ["hole_left_front", "hole_left_back", "hole_right_front", "hole_right_back"]
COMPONENTS_SPEECH = {
    "hole_left_front": "front left hole",
    "hole_left_back": "back left hole",
    "hole_right_front": "front right hole",
    "hole_right_back": "back right hole"
}
STATES = ["hole_empty", "hole_green", "hole_gold"]
STATES_SPEECH = {
    "hole_empty": "undefined",
    "hole_green": "green washer",
    "hole_gold": "gold washer"
}


class CarModel:
    def __init__(self):
        self.components = OrderedDict()
        for c in self.components:
            self.components[c](Component(c, STATES))

        self.comp_index = 0

    def check_update(self, component, state):
        intended_comp, intended_state = self.next_instruction()
        if intended_comp is None and intended_state is None:
            return "no_change", {}

        before_comp, before_state = self.current_instruction()
        out_data = {"before_comp": before_comp,
                    "before_state": before_state,
                    "intended_comp": intended_comp,
                    "intended_state": intended_state,
                    "actual_comp": component,
                    "actual_state": state}

        update = self.components[component].update_state(state)

        return update, out_data


    def next_instruction(self):
        check = self.check_component_index()
        if check is None:
            return None, None
        elif check == "incremented":
            return self.next_instruction()

        comp = self.components[self.name_from_index(self.comp_index)]

        return comp, comp.next_step()

    def current_instruction(self):
        check = self.check_component_index()
        if check is None:
            return None, None
        elif check == "incremented":
            return self.current_instruction()

        comp = self.components[self.name_from_index(self.comp_index)]

        return comp, comp.current_step()

    def check_component_index(self):
        if self.comp_index >= len(self.components.keys()) - 1:
            return None

        if self.components[self.name_from_index(self.comp_index)].finished():
            self.comp_index += 1
            return "incremented"

    def name_from_index(self, index):
        return list(self.components.keys())[index]


class Component:
    def __init__(self, name, states):
        self.name = name
        self.states = states
        self.current_index = 0

    def finished(self):
        return self.current_index == len(self.states) - 1

    def current_step(self):
        return self.states[self.current_index]

    def next_step(self):
        if self.current_index >= len(self.states) - 1:
            return None

        return self.states[self.current_index + 1]

    def update_state(self, state):
        current_state = self.current_step()
        if state == current_state:
            return "no_change"

        expected_state = self.next_step()
        if expected_state is None:
            return "no_change"
        if expected_state == state:
            self.current_index += 1
            return "next_step"
        else:
            for i, item in enumerate(self.states):
                if state == item:
                    self.current_index = i
            return "back_step"

