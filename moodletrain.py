from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from train import *
from generator import *
from val import *

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity
)

# Define ModelCheckpoint callback
model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras', # Path to save the model file
    monitor='val_loss',          # Monitor the validation loss
    save_best_only=True,         # Save only the best model (based on 'val_loss')
    save_weights_only=False,     # Save the full model (architecture + weights)
    mode='min'                   # The monitored quantity ('val_loss') should be minimized
)

# Define the number of epochs for training
epochs = 50 # A sufficient number of epochs, as EarlyStopping will stop it sooner if performance plateaus

print("Starting model training...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1  # Show training progress and validation loss

)

print("Model training complete. History object contains training metrics.")


#from tensorflow.keras.models import load_model
# model = load_model('best_model.keras')
# history = model.fit(train_generator, validation_data=val_generator, epochs=additional_epochs, callbacks=[early_stopping, model_checkpoint])
