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

class Dot(Element):
    tag = "circle"
    cant_have_children = True
    
    def __init__(self, *, x=0, y=0, color="cornflowerblue", size=3):
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
        )

class Plot(GroupElement):
    def __init__(self, padding=4):
        super().__init__()
        self.absolute_coordinates = dict()
        self.padding = padding
    
    def add_points(self, points, **attributes):
        for x,y in points:
            self.children.append(
                Dot(x=x, y=y, **attributes)
            )
        return self
    
    def generate_children(self):
        inner_as_proportion = (100 - (self.padding*2))/100.0
        x_stats = stats(each.x for each in self.children if each.x is not None)
        y_stats = stats(each.y for each in self.children if each.y is not None)
        new_children = []
        for each in self.children:
            if hasattr(each, "transformed") and callable(getattr(each, "transformed")):
                # normalize the values
                new_child = each.transformed(
                    translate_x=-x_stats.min,
                    translate_y=-y_stats.min,
                ).transformed(
                    scale_x=1/x_stats.range,
                    scale_y=1/y_stats.range,
                # then add padding
                ).transformed(
                    scale_x=inner_as_proportion,
                    scale_y=inner_as_proportion,
                ).transformed(
                    translate_x=self.padding,
                    translate_y=self.padding,
                )
                new_children.append(new_child)
            else:
                new_children.append(each)
        return new_children
    
    
    # TODO: background
    # TODO: y lines
    # TODO: x lines
    # TODO: x label
    # TODO: y label
    # TODO: title
    
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
