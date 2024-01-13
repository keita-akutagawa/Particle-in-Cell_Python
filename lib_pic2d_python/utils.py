def get_memory_used(x, unit):
    used_memory = x.__sizeof__()

    if unit == "KB":
        return round(used_memory/1024, 3)
    elif unit == "MB":
        return round(used_memory/1024**2, 3)
    elif unit == "GB":
        return round(used_memory/1024**3, 3)
    else:
        print("select unit KB or MB or GB.")
        return 0 
    
    