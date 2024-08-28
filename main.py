import cv2                  #version- 4.9.0
import numpy as np          #version- 1.23.5


def find_lines_and_angle(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 300, threshold=50, minLineLength=100, maxLineGap=100)

    if lines is not None:
        line_segments = [((x1, y1), (x2, y2)) for line in lines for x1, y1, x2, y2 in line]
        
        # Calculate lengths of the line segments
        lengths = [np.linalg.norm(np.array((x2, y2)) - np.array((x1, y1))) for (x1, y1), (x2, y2) in line_segments]
        
        # Sort the line segments by length
        sorted_lines = [line for _, line in sorted(zip(lengths, line_segments), reverse=True)]
        
        # Select the two longest lines
        if len(sorted_lines) >= 2:
            line1 = sorted_lines[0]
            line2 = sorted_lines[1]

            # Draw the lines
            cv2.line(image, line1[0], line1[1], (0, 255, 0), 2)
            cv2.line(image, line2[0], line2[1], (0, 255, 0), 2)

            # Calculate the intersection point of the lines
            intersection_point = calculate_intersection(line1, line2)
            
            if intersection_point is not None:
                # Mark the intersection point with a red circle
                cv2.circle(image, intersection_point, 5, (0, 0, 255), -1)
            
                # Label the points
                label_points(image, line1[0], 'A')
                label_points(image, line1[1], 'B')
                label_points(image, line2[0], 'C')
                label_points(image, line2[1], 'D')

                
                # Calculate the angle between the two lines
                angle = calculate_angle_between_lines(line1, line2)

                

                # Display the image with detected lines and intersection point
                cv2.imshow('Detected Pencils', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Save the image with detected lines and intersection point
                cv2.imwrite('detected_pencils_with_lines.jpg', image)

                return line1, line2, angle
            else:
                print("Lines are parallel; no intersection point.")
                return None
        else:
            print("Not enough lines detected")
            return None
    else:
        print("No lines detected")
        return None

def calculate_intersection(line1, line2):
    x1, y1, x2, y2 = *line1[0], *line1[1]
    x3, y3, x4, y4 = *line2[0], *line2[1]

    # Solve the linear system to find the intersection point
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if denominator != 0:
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return int(px), int(py)
    else:
        return None

def calculate_length(line1,line2):
    x1, y1, x2, y2 = *line1[0], *line1[1]
    x3, y3, x4, y4 = *line2[0], *line2[1]
     
    A=(x1,y1)
    B=(x2,y2)
    length_AB = np.linalg.norm(np.array(B) - np.array(A))
    print(f"Length of pencil A: {length_AB}")
    
    C=(x3,y3)
    D=(x4,y4)
    length_CD = np.linalg.norm(np.array(D) - np.array(C))
    print(f"Length of pencil B: {length_CD}")


def calculate_angle_between_lines(line1, line2):
    # Extract points from the lines
    x1, y1, x2, y2 = *line1[0], *line1[1]
    x3, y3, x4, y4 = *line2[0], *line2[1]
    
    # Calculate direction vectors
    vector1 = np.array([x2 - x1, y2 - y1])
    vector2 = np.array([x4 - x3, y4 - y3])
    
    # Calculate the dot product of the vectors
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Handle floating-point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    
    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def label_points(image, point, label):
    cv2.putText(image, label, (point[0] + 5, point[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(image, point, 5, (255, 0, 0), -1)

# Path to your image
image_path = '2_pencil.png'  # Replace with the actual path to your image
result = find_lines_and_angle(image_path)
if result:
    line1, line2, angle = result
    print(f"Endpoints of the pencil A: {line1}")
    print(f"Endpoints of the pencil B: {line2}")
    calculate_length(line1,line2)
    print(f"Angle between the pencils: {angle:.2f} degrees")
