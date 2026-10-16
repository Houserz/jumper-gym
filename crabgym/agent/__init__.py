"""The ER-led run: Gemini Robotics-ER 2 decides every step, and this package
gives it the operations to do so (`tools`), runs what takes time as actions
it can wait on (`actions`), remembers the phase it declared (`mission`),
keeps what the robot knows (`world_model`), reviews each call (`guard`), and
holds the conversation (`loop`), rendered once for the model and for the
run's record (`conversation`). `emulator` stands in for ER without a key.
"""
