batch_size = 32
epochs = 50

history = model.fit(datagen.flow(train_data, train_labels, batch_size=batch_size),
                    validation_data=(test_data, test_labels),
                    steps_per_epoch=len(train_data) // batch_size,
                    epochs=epochs)
# model evalution
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# save model
model.save("skin_tone_classification_model_v2.h5")