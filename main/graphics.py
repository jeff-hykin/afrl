import svgwrite
from svgwrite import cm, mm, percent
from tools import stats
import file_system_py as FS
from super_map import LazyDict

def html_escape_quoted_attribute(string):
    return string.replace('&', '&amp;').replace('"', '&quot;').replace("'", '&#39;').replace('<', '&lt;').replace('>', '&gt;')

def html_attribute(key, value):
    key = key.replace("_","-")
    return f''' {key}="{html_escape_quoted_attribute(str(value))}" '''

def indent(string, amount=4):
    indent_string = " "*amount
    return indent_string + string.replace("\n", f"\n{indent_string}")

class Element:
    tag = None
    cant_have_children = False
    attributes = {}
    x    = None
    y    = None
    size = None
    indent_size = 4
    def __init__(self):
        self.children = []
        # for each_key, each_value in attributes.items():
        #     setattr(self, each_key, each_value)
    
    def add(self, *elements):
        self.children += elements
        return self
    
    def remove(self, element):
        self.children = [ each for each in self.children if each != element ]
        return self
    
    @property
    def element_head_str_start(self):
        string = f"<{self.tag}"
        for each_key, each_value in self.attributes.items():
            string += html_attribute(each_key, each_value)
        return string
    
    @property
    def element_head_str(self):
        return self.element_head_str_start + ">\n"
    
    @property
    def children_str(self):
        string = ""
        for each in self.generate_children():
            string += indent(str(each), self.indent_size) + "\n"
        return string
    
    @property
    def element_tail_str(self):
        return f"</{self.tag}>"
    
    def generate_children(self):
        return self.children
    
    def __str__(self):
        # 
        # <tag />
        # 
        if self.cant_have_children:
            return self.element_head_str_start +  "/>"
        # 
        # <top>
        #     middle
        # </bottom>
        # 
        else:
            string = ""
            string += self.element_head_str
            string += self.children_str
            string += self.element_tail_str
            return string

class Document(Element):
    def __init__(self, ):
        super().__init__()
        self.tag = "svg"
        self.indent_size = Element.indent_size * 3
        
    @property
    def element_head_str(self):
        string = ""
        string += '''<?xml version="1.0" encoding="utf-8" ?>\n''' 
        string += '''<svg baseProfile="full" height="100%" version="1.1" viewBox="0,0,100,100" width="100%" xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink" >\n''' 
        string += '''    <defs />\n''' 
        string += '''    <g transform="scale(1,-1)" transform-origin="0 0">\n'''   # these two lines flip the y axis to be like math coordinates 
        string += '''        <svg style="overflow:visible;" x="0%" y="-100%">\n''' # 
        return string
    
    @property
    def element_tail_str(self):
        string = ""
        string += '''        </svg>\n''' 
        string += '''    </g>\n''' 
        string += '''</svg>\n'''   
        return string
    
    def save(self, to):
        FS.write(data=self.__str__(), to=to)

class SvgElement(Element):
    tag = "svg"
    def __init__(self, **attributes):
        super().__init__()
        self.attributes = attributes

class GroupElement(Element):
    tag = "g"
    def __init__(self, **attributes):
        super().__init__()
        self.attributes = attributes

class RectangleElement(Element):
    tag = "rect"
    cant_have_children = True
    def __init__(self, **attributes):
        super().__init__()
        self.attributes = attributes

class Rectangle(Element):
    tag = "rect"
    cant_have_children = True
    def __init__(self, *, x=0, y=0, color="cornflowerblue", width=10, height=10, roundedness=0):
        super().__init__()
        self.x           = x
        self.y           = y
        self.color       = color
        self.width       = width
        self.height      = height
        self.roundedness = roundedness
        self.size   = 1
    
    @property
    def attributes(self):
        return {
            "x": f"{self.x}%",
            "y": f"{self.y}%",
            "width": f"{self.width * self.size}%",
            "height": f"{self.height * self.size}%",
            "rx": f"{self.roundedness}%",
            "fill": str(self.color),
            "stroke-width": "0",
        }
    
    def transformed(self, scale_x=1, scale_y=1, scale_size=1, translate_x=0, translate_y=0,):
        rectangle = Rectangle(
            x=(self.x*scale_x)+translate_x,
            y=(self.y*scale_y)+translate_y,
            color=self.color,
            width=self.width,
            height=self.height,
            roundedness=self.roundedness,
        )
        rectangle.size = self.size*scale_size
        return rectangle

class Line(Element):
    tag = "line"
    cant_have_children = True
    def __init__(self, *, x1, y1, x2, y2, width=1, color="cornflowerblue"):
        super().__init__()
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.width = width
        self.color = color
        self.size  = 1
    
    @property
    def attributes(self):
        return {
            "x1": f"{self.x1}%",
            "y1": f"{self.y1}%",
            "x2": f"{self.x2}%",
            "y2": f"{self.y2}%",
            "stroke": self.color,
            "stroke-width": str(self.width),
        }
    
    def transformed(self, scale_x=1, scale_y=1, scale_size=1, translate_x=0, translate_y=0,):
        line = Line(
            x1=(self.x1*scale_x)+translate_x,
            y1=(self.y1*scale_y)+translate_y,
            x2=(self.x2*scale_x)+translate_x,
            y2=(self.y2*scale_y)+translate_y,
            width=self.width,
            color=self.color,
        )
        # TODO: size would need to adjust the x1,y1,x2,y2 and should do so by expanding around the center point
        #       thats a chunk of math I'll do later
        # line.size = self.size*scale_size
        return line

class Dot(Element):
    tag = "circle"
    cant_have_children = True
    
    def __init__(self, *, x=0, y=0, color="cornflowerblue", size=1.2):
        super().__init__()
        self.x     = x
        self.y     = y
        self.color = color
        self.size  = size
    
    @property
    def attributes(self):
        return {
            "cx": f"{self.x}%",
            "cy": f"{self.y}%",
            "r":  f"{self.size}%",
            "fill": str(self.color),
            "stroke-width": "0",
        }
    
    def transformed(self, scale_x=1, scale_y=1, scale_size=1, translate_x=0, translate_y=0,):
        return Dot(
            x=(self.x*scale_x)+translate_x,
            y=(self.y*scale_y)+translate_y,
            size=self.size*scale_size,
            color=self.color,
        )

class Plot(GroupElement):
    def __init__(self, padding=5, background_color="whitesmoke", roundedness=2):
        super().__init__()
        self.padding = padding
        self.background_color = background_color
        self.roundedness = roundedness
    
    def add_points(self, points, **attributes):
        for x,y in points:
            self.children.append(
                Dot(x=x, y=y, **attributes)
            )
        return self
    
    def generate_children(self):
        children = self.children
        padding = self.padding
        new_children = []
        
        # 
        # background
        # 
        new_children.append(
            Rectangle(x=0, y=0, width=100, height=100, color=self.background_color, roundedness=self.roundedness)
        )
        
        padding_proportion = padding/100
        inner_as_proportion = (100 - (padding*2))/100.0
        x_stats = stats(each.x for each in children if each.x is not None)
        y_stats = stats(each.y for each in children if each.y is not None)
        
        # 
        # vertical lines
        # 
        number_of_vertical_lines = 5
        percent = 100/(number_of_vertical_lines+1)
        for number in range(1,number_of_vertical_lines+1):
            new_children.append(
                Line(
                    x1=percent * number,
                    y1=self.padding/2,
                    x2=percent * number,
                    y2=100-(self.padding/2),
                    width=0.2,
                    color="lightgray",
                )
            )
        # 
        # horizontal lines
        # 
        number_of_horizontal_lines = 5
        percent = 100/(number_of_horizontal_lines+1)
        for number in range(1,number_of_horizontal_lines+1):
            new_children.append(
                Line(
                    x1=self.padding/2,
                    y1=percent * number,
                    x2=100-(self.padding/2),
                    y2=percent * number,
                    width=0.2,
                    color="lightgray",
                )
            )
        
        # 
        # normal children
        # 
        for each in children:
            if hasattr(each, "transformed") and callable(getattr(each, "transformed")):
                # normalize the values
                new_child = each.transformed(
                    translate_x=-x_stats.min,
                    translate_y=-y_stats.min,
                ).transformed(
                    scale_x=100/x_stats.range,
                    scale_y=100/y_stats.range,
                )
                # then squish to fit inside the padding
                new_child = new_child.transformed(
                    scale_x=inner_as_proportion,
                    scale_y=inner_as_proportion,
                )
                # then add the padding
                new_child = new_child.transformed(
                    translate_x=padding,
                    translate_y=padding,
                )
                new_children.append(new_child)
            else:
                new_children.append(each)
        
        # TODO: x label
        # TODO: y label
        # TODO: title
        
        return new_children
    
    
    
    def save(self, *args, **kwargs):
        svg = Document().add(self)
        return svg.save(*args, **kwargs)

plot = Plot().add_points([[1,1],[2,2.5],[3,5]]).save("test.svg")



def basic_shapes(name):
    drawing = svgwrite.Drawing(filename=name, debug=True)
    hlines = drawing.add(drawing.g(id='hlines', stroke='green'))
    for y in range(20):
        hlines.add(drawing.line(start=(2*cm, (2+y)*cm), end=(18*cm, (2+y)*cm)))
    vlines = drawing.add(drawing.g(id='vline', stroke='blue'))
    for x in range(17):
        vlines.add(drawing.line(start=((2+x)*cm, 2*cm), end=((2+x)*cm, 21*cm)))
    shapes = drawing.add(drawing.g(id='shapes', fill='red'))

    # set presentation attributes at object creation as SVG-Attributes
    circle = drawing.circle(center=(15*cm, 8*cm), r='2.5cm', stroke='blue', stroke_width=3)
    circle['class'] = 'class1 class2'
    shapes.add(circle)
