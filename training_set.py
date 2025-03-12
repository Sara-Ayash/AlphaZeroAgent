from typing import List
from pydantic import BaseModel


class TrainingValue(BaseModel):
    game_state: List[int]
    winner: int = None
    policy_target: List[float]
 
    def model_dump(self):
        return f"{(self.game_state, (self.policy_target, self.winner))}"
          
    def set_winner(self, winner):
        self.winner = winner


class TrainingData(BaseModel):
    training_set: List[TrainingValue] = []

    def add_training_value(self, training_value: TrainingValue):
        self.training_set.append(training_value)

    def extend_training_set(self, training_set_list):
        self.training_set.extend(training_set_list)

    def model_dump(self):
        return [tv.model_dump() for tv in self.training_set]

    def save_to_text_file(self, filename: str):
        with open(filename, "a") as f:
            for line in self.model_dump():
                f.write(line + "\n")  
 



        
