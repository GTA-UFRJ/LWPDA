from LWPDA import lwpda

for threshold in range(0, 11):
    a = lwpda('yolov8n-seg.pt', verbose = False, show = False, threshold = threshold*10)
    a.experiments('/home/guardai/antunes/IMAGENET/all_videos/', f'/home/guardai/antunes/newTests/{threshold}/bb/')

