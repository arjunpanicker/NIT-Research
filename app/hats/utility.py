import fasttext

_deviceList = ['ac', 'tv', 'fan', 'light', 'geyser']

def device_exists(command: str, ft_model) -> bool:
    '''Checks whether the given string contains one of the many devices
    supported by the model. 
    '''
    # Get the 4 nearest words to each device and create a dictionary
    top_nearest_words_to_devices = {}
    for device in _deviceList:
        top_nearest_words_to_devices[device] = []
        nearestWords = ft_model.get_nearest_neighbors(device, k=4)
        top_nearest_words_to_devices[device].extend([word for _, word in nearestWords])

    for word in command:
        for device in _deviceList:
            if word in top_nearest_words_to_devices[device]:
                return True
    
    return False