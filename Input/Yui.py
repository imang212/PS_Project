# Version 0.1
# Changelog
# |   Version 0.1
# |   |   Initial creation
# |   Version 0.1.1
# |   |   Added Button and Switch
# |   Version 0.1.2
# |   |   Bug Fixes:
# |   |   |   text displayed in the wrong place
# |   Version 0.1.3
# |   |   Yui:
# |   |   |   Added Yui.local_subtree_bounds and Yui.global_subtree_bounds
# |   |   Stack:
# |   |   |   Added Stack
# |   |   Graphics:
# |   |   |   Added Graphics.text_width
# |   |   TextField:
# |   |   |   Added TextField
# |   |   Mouse & MouseEvent:
# |   |   |   Added Mouse.pass_event, MosueEvent.to_world and MouseEvent.pass_to
# |   |   Switch:
# |   |   |   Added radio feature
# |   Version 0.1.4
# |   |   Fixed TextField
# |   |   Fixed Graphics.ellipse()
# |   |   Fixed Yui.draw()
# |   |   Added Slider
# |   Version 0.1.5
# |   |   Added TabView
# |   |   Implemented enabling
# |   |   Minor bug fixes
# TODO:
# |   Resizable
# |   GPU drawing (with GL)
# |   Basic UI elements
# |   UI elements: Stack, TextField, Slider, InfiniteCanvas
# |   Fix align

from __future__ import annotations
import time
from typing import Iterable, Iterator
import numpy as np
import pygame
from abc import ABC, abstractmethod
import colorsys
import threading
from enum import Enum

# === Utils ===

class Matrix2D(np.ndarray):
    """
    A 3x3 transformation matrix for 2D graphics.
    This class represents a 3x3 matrix used for transformations in 2D space.
    It supports operations like translation, rotation, scaling, and point transformation.
    It is a subclass of numpy.ndarray, allowing for matrix operations.
    
    Methods:
        - __new__: Creates a new 3x3 transformation matrix.
        - transform_point: Applies the transformation to a 2D point (x, y).
        - identity: Returns the identity matrix.
        - translation: Returns a translation matrix.
        - rotation: Returns a rotation matrix given an angle in radians.
        - scaling: Returns a scaling matrix.
        - decompose: Decomposes the matrix into translation, rotation, and scale.
        - translate: Returns a new matrix with translation applied after this one.
        - rotate: Returns a new matrix with rotation applied after this one.
        - scale: Returns a new matrix with scaling applied after this one.
        - invert: Returns the inverse of the transformation matrix.
    """
    
    def __new__(cls, input_array=None) -> 'Matrix2D':
        """
        Creates a new 3x3 transformation matrix.
        The matrix is initialized to the identity matrix if no input_array is provided.
        The input_array should be a 3x3 array-like structure (list, tuple, or numpy array).
        If the input_array is not 3x3, a ValueError is raised.
        The matrix is stored as a numpy array with float type.
        The class is a subclass of numpy.ndarray, allowing for matrix operations.

        Args:
            input_array (np.ndarray, optional): A 3x3 array-like structure to initialize the matrix. If None, initializes to the identity matrix.

        Raises:
            ValueError: If the input_array is not a 3x3 array.

        Returns:
            Matrix2D: An instance of Matrix2D initialized with the provided input_array or the identity matrix.
        """
        if input_array is None:
            input_array = np.identity(3, dtype=float)
        obj = np.asarray(input_array, dtype=float).view(cls)
        if obj.shape != (3, 3):
            raise ValueError("Matrix2D must be a 3x3 array.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # No additional attributes for now

    def __matmul__(self, other):
        if isinstance(other, Vector2D):
            result = super().__matmul__(other)
            return Vector2D(result[0, 0], result[1, 0])
        return super().__matmul__(other)
    
    def __rmatmul__(self, other):
        if isinstance(other, Matrix2D):
            result = other @ self
            return Vector2D(result[0, 0], result[1, 0])
        return NotImplemented

    
    @classmethod
    def identity(cls):
        """
        Returns the identity matrix for 2D transformations.
        """
        return cls(np.identity(3))

    @classmethod
    def translation(cls, tx:float, ty: float) -> 'Matrix2D':
        """
        Returns a translation matrix that translates points by (tx, ty).

        Args:
            tx (float): x translation
            ty (float): y translation

        Returns:
            Matrix2D: A translation matrix that translates points by (tx, ty).
        """
        return cls([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    @classmethod
    def rotation(cls, theta_rad: float) -> 'Matrix2D':
        """
        Returns a rotation matrix for a given angle in radians.

        Args:
            theta_rad (float): Angle in radians to rotate points.

        Returns:
            Matrix2D: A rotation matrix that rotates points by theta_rad radians.
        """
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        return cls([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    @classmethod
    def scaling(cls, sx:float, sy:float) -> 'Matrix2D':
        """
        Returns a scaling matrix that scales points by (sx, sy).

        Args:
            sx (float): x scaling factor
            sy (float): x scaling factor

        Returns:
            Matrix2D: A scaling matrix that scales points by (sx, sy).
        """
        return cls([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1]
        ])

    def __repr__(self):
        return f"Matrix2D(\n{super().__repr__()}\n)"
    
    def decompose(self) -> tuple:
        """
        Decomposes the matrix into translation, rotation, and scale components.

        Returns:
            tuple: A tuple containing:
                - translation (tx, ty): The translation components.
                - theta (float): The rotation angle in radians.
                - scale (sx, sy): The scaling factors in x and y directions.
        """
        a, c, tx = self[0]
        b, d, ty = self[1]

        # Extract translation directly
        translation = (tx, ty)

        # Compute scale
        sx = np.hypot(a, b)
        sy = np.hypot(c, d)

        # Normalize to remove scale from rotation matrix
        if sx != 0: a_n, b_n = a / sx, b / sx
        else:       a_n, b_n = a, b
        if sy != 0: c_n, d_n = c / sy, d / sy
        else:       c_n, d_n = c, d

        # Compute rotation from normalized matrix (assumes uniform scaling + no skew)
        theta = np.arctan2(b_n, a_n)

        return translation, theta, (sx, sy)
    
    def translate(self, tx:float, ty:float) -> 'Matrix2D':
        """
        Returns a new matrix with translation applied *after* this one.

        Args:
            tx (float): x translation
            ty (float): y translation

        Returns:
            Matrix2D: A new transformation matrix with translation applied after this one.
        """
        return self @ Matrix2D.translation(tx, ty)

    def rotate(self, theta_rad:float) -> 'Matrix2D':
        """
        Returns a new matrix with rotation applied *after* this one.

        Args:
            theta_rad (float): Angle in radians to rotate points.

        Returns:
            Matrix2D: A new transformation matrix with rotation applied after this one.
        """
        return self @ Matrix2D.rotation(theta_rad)

    def scale(self, sx:float, sy:float) -> 'Matrix2D':
        """
        Returns a new matrix with scaling applied *after* this one.

        Args:
            sx (float): x scaling factor
            sy (float): y scaling factor

        Returns:
            Matrix2D: A new transformation matrix with scaling applied after this one.
        """
        return self @ Matrix2D.scaling(sx, sy)

    def invert(self) -> 'Matrix2D':
        """
        Returns the inverse of the transformation matrix.

        Raises:
            ValueError: If the matrix is not invertible.

        Returns:
            Matrix2D: The inverse of the transformation matrix.
        """
        try:
            inv = np.linalg.inv(self)
            return Matrix2D(inv)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible.")

class Vector2D(np.ndarray):
    """
    A 2D vector represented as a 3x1 column vector.
    This class represents a 2D vector in homogeneous coordinates, allowing for
    transformations using a 3x3 Matrix2D. It supports basic vector operations
    such as addition, subtraction, scaling, and transformation by a Matrix2D.
    It is a subclass of numpy.ndarray, allowing for array-like operations.
    
    Attributes:
        x (float): The x component of the vector.
        y (float): The y component of the vector.
    Methods:
        - __new__: Creates a new Vector2D instance.
        - transform: Applies a Matrix2D transformation and returns a new Vector2D.
        - magnitude: Returns the distance from another vector or the origin.
        - heading: Returns the angle (in radians) from the origin or another vector.
        - __add__, __sub__, __mul__, __truediv__, __neg__: Basic vector operations.
        - swizzle: Returns a new Vector2D based on a pattern string.
    """
    def __new__(cls, x=0.0, y=0.0) -> 'Vector2D':
        ar = [[x], [y], [1.0]]
        data = np.array(ar, dtype=float)
        obj = np.asarray(data).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None: return

    @property
    def x(self) -> float:
        """The x component of the vector."""
        return self[0, 0]

    @x.setter
    def x(self, value) -> None:
        """Sets the x component of the vector."""
        self[0, 0] = value

    @property
    def y(self) -> float:
        """The y component of the vector."""
        return self[1, 0]

    @y.setter
    def y(self, value) -> None:
        """Sets the y component of the vector."""
        self[1, 0] = value

    def transform(self, matrix: Matrix2D) -> 'Vector2D':
        """
        Applies a Matrix2D transformation to this vector and returns a new Vector2D.

        Args:
            matrix (Matrix2D): The transformation matrix to apply.

        Returns:
            Vector2D: A new Vector2D that is the result of applying the transformation.
        """
        result = matrix @ self
        return Vector2D(result[0, 0], result[1, 0])

    def magnitude(self, origin:'Vector2D'=None) -> float:
        """
        Returns the distance from this vector to another vector or the origin.

        Args:
            origin (Vector2D, optional): Another vector to measure distance from. If None, measures from the origin (0, 0).

        Returns:
            float: The distance from this vector to the origin or another vector.
        """
        dx, dy = self.x, self.y
        if origin is not None:
            dx -= origin.x
            dy -= origin.y
        return np.hypot(dx, dy)

    def heading(self, origin:'Vector2D'=None) -> float:
        """
        Returns the angle (in radians) from this vector to another vector or the origin.

        Args:
            origin (Vector2D, optional): Another vector to measure angle from. If None, measures from the origin (0, 0).

        Returns:
            float: The angle in radians from this vector to the origin or another vector.
        """
        dx, dy = self.x, self.y
        if origin is not None:
            dx -= origin.x
            dy -= origin.y
        return np.arctan2(dy, dx)

    def __add__(self, other:'Vector2D') -> 'Vector2D':
        """
        Adds another Vector2D to this vector and returns a new Vector2D.

        Args:
            other (Vector2D): Another vector to add.

        Returns:
            Vector2D: A new Vector2D that is the sum of this vector and the other vector.
        """
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other:'Vector2D') -> 'Vector2D':
        """
        Subtracts another Vector2D from this vector and returns a new Vector2D.

        Args:
            other (Vector2D): Another vector to subtract.

        Returns:
            Vector2D: A new Vector2D that is the difference of this vector and the other vector.
        """
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Multiplies this vector by a scalar or another Vector2D and returns a new Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value or another Vector2D to multiply with.

        Returns:
            Vector2D: A new Vector2D that is the product of this vector and the scalar or vector.
        """
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Allows scalar multiplication with Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value or another Vector2D to multiply with.

        Returns:
            Vector2D: A new Vector2D that is the product of the scalar and this vector.
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Divides this vector by a scalar and returns a new Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value to divide by.

        Returns:
            Vector2D: A new Vector2D that is the result of dividing this vector by the scalar.
        """
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Vector2D':
        """
        Negates this vector and returns a new Vector2D.

        Returns:
            Vector2D: A new Vector2D that is the negation of this vector.
        """
        return Vector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Vector2D.

        Returns:
            str: A string representation of the Vector2D in the format "Vector2D(x=..., y=...)".
        """
        return f"Vector2D(x={self.x}, y={self.y})"
        
    def __rmatmul__(self, matrix:Matrix2D) -> 'Vector2D':
        """
        Allows matrix multiplication with a Matrix2D on the left side.

        Args:
            matrix (Matrix2D): The transformation matrix to apply.

        Returns:
            Vector2D: A new Vector2D that is the result of applying the transformation matrix to this vector.
            
        Raises:
            NotImplemented: If the left operand is not a Matrix2D.
        """
        if isinstance(matrix, Matrix2D):
            result = super().__matmul__(matrix)
            return Vector2D(result[0, 0], result[0, 1])
        return NotImplemented
    
    def swizzle(self, pattern:str) -> 'Vector2D':
        """
        Returns a new Vector2D based on a swizzle pattern string.
        The pattern can contain:
            - 'x' for x component
            - 'y' for y component
            - '0' for zero
            - '1' for one
            - 'n1' for negative one
            - 'nx' for negative x component
            - 'ny' for negative y component
        The pattern must be exactly 2 characters long, and each character must be one of the above tokens.
        
        Args:
            pattern (str): A 2-character string representing the swizzle pattern.

        Raises:
            ValueError: If the pattern is not exactly 2 characters long or contains invalid tokens.

        Returns:
            Vector2D: A new Vector2D created based on the swizzle pattern.
        """
        if len(pattern) != 2:
            raise ValueError("Swizzle pattern must be length 2.")

        def get_val(token):
            if token == 'x':
                return self.x
            elif token == 'y':
                return self.y
            elif token == '0':
                return 0.0
            elif token == '1':
                return 1.0
            elif token == 'n1':
                return -1.0
            elif token == 'nx':
                return -self.x
            elif token == 'ny':
                return -self.y
            else:
                raise ValueError(f"Invalid swizzle token: {token}")

        # We need to parse tokens: either 1 or 2 chars, so:
        # 'n1', 'nx', 'ny' are 2-char tokens
        # 'x', 'y', '0', '1' are 1-char tokens

        # Parse the pattern into tokens accordingly:
        tokens = []
        i = 0
        while i < len(pattern):
            # Check for 2-char tokens starting with 'n'
            if pattern[i] == 'n' and i + 1 < len(pattern):
                tokens.append(pattern[i:i+2])
                i += 2
            else:
                tokens.append(pattern[i])
                i += 1

        if len(tokens) != 2:
            raise ValueError("Swizzle pattern must resolve to exactly 2 tokens.")

        x_val = get_val(tokens[0])
        y_val = get_val(tokens[1])
        return Vector2D(x_val, y_val)
    
    def to_tuple(self):
        """
        Returns the (x, y) components of the vector as a tuple.

        Returns:
            tuple: A tuple (x, y) representing the vector components.
        """
        return (self.x, self.y)
    @classmethod
    def random(cls, low:float=0.0, high:float=1.0) -> 'Vector2D':
        """
        Returns a new Vector2D with random x and y components within the specified range.

        Args:
            low (float): The lower bound for the random values (inclusive).
            high (float): The upper bound for the random values (exclusive).

        Returns:
            Vector2D: A new Vector2D with random x and y components.
        """
        return cls(np.random.uniform(low, high), np.random.uniform(low, high))
    
    @classmethod
    def random_unit(cls) -> 'Vector2D':
        """
        Returns a new Vector2D with random x and y components that form a unit vector.

        Returns:
            Vector2D: A new Vector2D with random direction but unit length.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        return cls(np.cos(angle), np.sin(angle))
    
    @classmethod
    def polar(cls, radius:float, angle_rad:float) -> 'Vector2D':
        """
        Returns a new Vector2D from polar coordinates.

        Args:
            radius (float): The distance from the origin.
            angle_rad (float): The angle in radians.

        Returns:
            Vector2D: A new Vector2D representing the point in Cartesian coordinates.
        """
        return cls(radius * np.cos(angle_rad), radius * np.sin(angle_rad))

class Color(pygame.Color):
    def __new__(cls, r=0, g=0, b=0, a=255):
        """
        Creates a new Color instance with the specified RGB(A) values.
        
        Args:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).
            a (int, optional): Alpha component (0-255). Defaults to 255 (opaque).
        
        Returns:
            Color: A new Color instance.
        """
        r = int(max(0, min(255, r)))
        g = int(max(0, min(255, g)))
        b = int(max(0, min(255, b)))
        a = int(max(0, min(255, a)))
        return super().__new__(cls, r, g, b, a)
    
    def __repr__(self):
        """
        Returns a string representation of the Color instance.
        
        Returns:
            str: A string representation of the Color in the format "Color(r, g, b, a)".
        """
        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"
    
    def __str__(self):
        return super().__str__()
    
    def to_tuple(self) -> tuple:
        """
        Returns the color as a tuple (r, g, b, a).
        
        Returns:
            tuple: A tuple containing the RGBA components of the color.
        """
        return (self.r, self.g, self.b, self.a)
    
    def to_hex(self) -> str:
        """
        Returns the color as a hexadecimal string.
        
        Returns:
            str: A hexadecimal string representation of the color, e.g., "#RRGGBBAA".
        """
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        """
        Creates a Color instance from a hexadecimal string.
        
        Args:
            hex_str (str): A hexadecimal string in the format "#RRGGBB" or "#RRGGBBAA".
        
        Returns:
            Color: A new Color instance created from the hexadecimal string.
        
        Raises:
            ValueError: If the hex_str is not in a valid format.
        """
        if hex_str.startswith('#'):
            hex_str = hex_str[1:]
        if len(hex_str) == 6:
            return cls(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        elif len(hex_str) == 8:
            return cls(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), int(hex_str[6:8], 16))
        else:
            raise ValueError("Hex string must be in format '#RRGGBB' or '#RRGGBBAA'.")

    @classmethod
    def from_hsb(cls, h: float=0, s: float=0, b: float=0, a: int = 255) -> 'Color':
        """
        Creates a Color instance from HSB/HSV values.

        Args:
            h (float): Hue, in range [0, 1].
            s (float): Saturation, in range [0, 1].
            b (float): Brightness/Value, in range [0, 1].
            a (int, optional): Alpha component (0-255). Defaults to 255.

        Returns:
            Color: A new Color instance created from the HSB values.
        """
        r, g, b = colorsys.hsv_to_rgb(h / 255, s / 255, b / 255)
        return cls(int(r * 255), int(g * 255), int(b * 255), a)

# === Graphics === 

# Global flag to determine if OpenGL should be used
_use_gl = False  # Default to not using OpenGL
def use_gl(use:bool|None=None) -> None|bool:
    """
    Sets or gets the OpenGL usage flag for the graphics module.
    
    Args:
        use (bool, optional): If provided, sets the OpenGL usage flag. If None, returns the current flag.
    
    Returns:
        bool: The current OpenGL usage flag if no argument is provided.
    """
    global _use_gl
    if use is not None:
        _use_gl = use
    return _use_gl

# Also allows pixel access like a numpy array
class Graphics(pygame.surface.Surface):
    """
    A graphics surface that can be used for drawing shapes, images, and text.
    This class is a subclass of pygame.Surface and provides additional methods
    for drawing common shapes and handling transformations.
    
    Variables:
        width (int): The width of the surface.
        height (int): The height of the surface.
        
    """
    
    MODES = ["corner", "corners", "center", "radius"]
    
    def __init__(self, width:int, height:int) -> 'Graphics':
        """
        Creates a new Graphics surface with the specified width and height.
        
        Args:
            width (int): The width of the surface.
            height (int): The height of the surface.
        
        Returns:
            Graphics: A new Graphics instance.
        """
        
        super().__init__((width, height), pygame.SRCALPHA)
        
        # Add additional attributes
        self._transforms = [Matrix2D.identity()]
        
        self._fill_color = Color(0, 0, 0, 255)  # Default fill color
        self._stroke_color = Color(255, 255, 255, 255)
        self._stroke_width = 1  # Default stroke width
        self._texture = None  # Default texture is None, can be a Graphics selfect
        self._gradient = None  # Default gradient is None, can be a tuple of two colors
        
        self._rect_mode = 'corner'  # Default rectangle mode
        self._ellipse_mode = 'center'  # Default ellipse mode
        self._image_mode = 'corner'  # Default image mode
        
        self._text_align_x = 0  # Horizontal text alignment (0: left, 1: right)
        self._text_align_y = 0  # Vertical text alignment (0: top, 1: bottom)
        self._text_size = 12  # Default text size
        self._text_font_path = None
        self._text_font = None
        self._text_leading = 0  # Default text leading (line spacing)
        
        self._curve_detail = 100  # Default curve detail for bezier curves
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        # No additional attributes for now
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Graphics surface.
        
        Returns:
            str: A string representation of the Graphics surface in the format "Graphics(width, height)".
        """
        return f"Graphics({self.get_width()}, {self.get_height()})"
    
    # Properties
    @property
    def width(self) -> int:
        """
        Returns the width of the Graphics surface.
        
        Returns:
            int: The width of the surface.
        """
        return self.get_width()
    
    @property
    def height(self) -> int:
        """
        Returns the height of the Graphics surface.
        
        Returns:
            int: The height of the surface.
        """
        return self.get_height()
    
    @property
    def last_transform(self) -> Matrix2D:
        """
        Returns the last transformation matrix applied to the Graphics surface.
        
        Returns:
            Matrix2D: The last transformation matrix.
        """
        return self._transforms[-1]
    @last_transform.setter
    def last_transform(self, value:Matrix2D):
        self._transforms[-1] = value
    
    @staticmethod
    def _coordinates(mode:str, x1:float, y1:float, x2:float, y2:float) -> tuple:
        """
        Converts coordinates based on the specified mode.
        The modes can be:
            - 'corner': (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            - 'corners': (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            - 'center': (x1, y1) is the center, (x2, y2) is the size
            - 'radius': (x1, y1) is the center, (x2, y2) is the radius

        Args:
            mode (str): The mode for coordinate conversion. Can be 'corner', 'corners', 'center', or 'radius'.
            x1 (float)
            y1 (float)
            x2 (float)
            y2 (float)

        Raises:
            ValueError: If the mode is not one of the expected values.

        Returns:
            tuple: A tuple representing the coordinates in the format (x, y, width, height) based on the mode.
        """
        if mode == 'corner':
            # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            return (x1, y1, x2, y2)
        elif mode == 'corners':
            # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            return (x1, y1, x2 - x1, y2 - y1)
        elif mode == 'center':
            # (x1, y1) is the center, (x2, y2) is the size
            return (x1 - x2 / 2, y1 - y2 / 2, x2, y2)
        elif mode == 'radius':
            # (x1, y1) is the center, (x2, y2) is the radius
            return (x1 - x2, y1 - y2, x2 * 2, y2 * 2)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
    
    @property
    def rect_mode(self) -> str:
        """
        Returns the current rectangle mode.
        
        Returns:
            str: The current rectangle mode ('corner', 'corners', 'center', or 'radius').
        """
        return self._rect_mode
    
    @rect_mode.setter
    def rect_mode(self, mode:str) -> None:
        """
        Sets the rectangle mode for drawing rectangles.
        
        Args:
            mode (str): The rectangle mode to set. Can be 'corner', 'corners', 'center', or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['corner', 'corners', 'center', 'radius']:
            raise ValueError(f"Invalid rectangle mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
        self._rect_mode = mode
    
    @property
    def ellipse_mode(self) -> str:
        """
        Returns the current ellipse mode.
        
        Returns:
            str: The current ellipse mode ('center' or 'radius').
        """
        return self._ellipse_mode
    
    @ellipse_mode.setter
    def ellipse_mode(self, mode:str) -> None:
        """
        Sets the ellipse mode for drawing ellipses.
        
        Args:
            mode (str): The ellipse mode to set. Can be 'center' or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['center', 'radius']:
            raise ValueError(f"Invalid ellipse mode: {mode}. Must be 'center' or 'radius'.")
        self._ellipse_mode = mode
    
    @property
    def image_mode(self) -> str:
        """
        Returns the current image mode.
        
        Returns:
            str: The current image mode ('corner', 'corners', 'center', or 'radius').
        """
        return self._image_mode
    
    @image_mode.setter
    def image_mode(self, mode:str) -> None:
        """
        Sets the image mode for drawing images.
        
        Args:
            mode (str): The image mode to set. Can be 'corner', 'corners', 'center', or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['corner', 'corners', 'center', 'radius']:
            raise ValueError(f"Invalid image mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
        self._image_mode = mode
    
    
    @property
    def text_align_x(self) -> int:
        """
        Returns the current horizontal text alignment.
        
        Returns:
            int: The current horizontal text alignment (0: left, 1: right).
        """
        return self._text_align_x
    @text_align_x.setter
    def text_align_x(self, align:int) -> None:
        """
        Sets the horizontal text alignment.
        
        Args:
            align (int): The horizontal text alignment to set (0: left, 1: right).
        
        Raises:
            ValueError: If the alignment is not 0 or 1.
        """
        align = max(0, min(1, align))
        self._text_align_x = align
        
    @property
    def text_align_y(self) -> int:
        """
        Returns the current vertical text alignment.
        
        Returns:
            int: The current vertical text alignment (0: top, 1: bottom).
        """
        return self._text_align_y
    @text_align_y.setter
    def text_align_y(self, align:int) -> None:
        """
        Sets the vertical text alignment.
        
        Args:
            align (int): The vertical text alignment to set (0: top, 1: bottom).
        
        Raises:
            ValueError: If the alignment is not 0 or 1.
        """
        align = max(0, min(1, align))
        self._text_align_y = align
    
    @property
    def text_align(self) -> tuple[float, float]:
        """
        Returns the current text alignment as a tuple (horizontal, vertical).
        
        Returns:
            tuple[int, int]: A tuple representing the horizontal and vertical text alignment.
        """
        return (self._text_align_x, self._text_align_y)
    @text_align.setter
    def text_align(self, align:tuple[float, float]) -> None:
        """
        Sets the text alignment.
        
        Args:
            align (tuple[int, int]): A tuple representing the horizontal and vertical text alignment.
        
        Raises:
            ValueError: If the alignment is not a tuple of two integers (0 or 1).
        """
        if not isinstance(align, tuple) or len(align) != 2:
            raise ValueError("text_align must be a tuple of two integers (horizontal, vertical).")
        self._text_align_x = max(0, min(1, align[0]))
        self._text_align_y = max(0, min(1, align[1]))
    
    @property
    def text_font(self) -> str:
        """
        Returns the current font used for text rendering.
        
        Returns:
            pygame.font.Font: The current font object.
        """
        return self._text_font_path
    @text_font.setter
    def text_font(self, font:str) -> None:
        """
        Sets the font for text rendering.
        
        Args:
            font (pygame.font.Font): The font object to set. If None, resets to the default font.
        
        Raises:
            TypeError: If the font is not a pygame.font.Font instance.
        """
        if font is not None and not isinstance(font, str):
            raise TypeError("text_font must be a str instance or None.")
        self._text_font_path = font
        
        try:
            self._text_font = pygame.font.Font(self._text_font_path, self._text_size)
        except Exception as e:
            self._text_font = pygame.font.SysFont(self._text_font_path, self._text_size)
        
    
    @property
    def text_size(self) -> int:
        """
        Returns the current text size.
        
        Returns:
            int: The current text size.
        """
        return self._text_size
    @text_size.setter
    def text_size(self, size:int) -> None:
        """
        Sets the text size for rendering.
        
        Args:
            size (int): The size to set as the text size.
        
        Raises:
            ValueError: If the size is less than 1.
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("text_size must be an integer greater than or equal to 1.")
        self._text_size = size
        try:
            self._text_font = pygame.font.Font(self._text_font_path, self._text_size)
        except:
            self._text_font = pygame.font.SysFont(self._text_font_path, self._text_size)
    
    @property
    def text_leading(self) -> int:
        """
        Returns the current text leading (line spacing).
        
        Returns:
            int: The current text leading.
        """
        return self._text_leading
    @text_leading.setter
    def text_leading(self, leading:int) -> None:
        """
        Sets the text leading (line spacing) for rendering.
        
        Args:
            leading (int): The leading to set for text rendering.
        
        Raises:
            ValueError: If the leading is less than 0.
        """
        if not isinstance(leading, int) or leading < 0:
            raise ValueError("text_leading must be an integer greater than or equal to 0.")
        self._text_leading = leading
    
    
    @property
    def fill_color(self) -> Color:
        """
        Returns the current fill color.
        
        Returns:
            Color: The current fill color.
        """
        return self._fill_color
    @fill_color.setter
    def fill_color(self, color:Color) -> None:
        """
        Sets the fill color for drawing shapes.
        
        Args:
            color (Color): The color to set as the fill color.
        """
        if not isinstance(color, Color):
            raise TypeError("fill_color must be a Color instance.")
        self._fill_color = color
    
    @property
    def stroke_color(self) -> Color:
        """
        Returns the current stroke color.
        
        Returns:
            Color: The current stroke color.
        """
        return self._stroke_color
    @stroke_color.setter
    def stroke_color(self, color:Color) -> None:
        """
        Sets the stroke color for drawing shapes.
        
        Args:
            color (Color): The color to set as the stroke color.
        
        Raises:
            TypeError: If the color is not a Color instance.
        """
        if not isinstance(color, Color):
            raise TypeError("stroke_color must be a Color instance.")
        self._stroke_color = color
    
    @property
    def stroke_width(self) -> int:
        """
        Returns the current stroke width.
        
        Returns:
            int: The current stroke width.
        """
        return self._stroke_width
    @stroke_width.setter
    def stroke_width(self, width:int) -> None:
        """
        Sets the stroke width for drawing shapes.
        
        Args:
            width (int): The width to set as the stroke width.
        
        Raises:
            ValueError: If the width is less than 1.
        """
        if not isinstance(width, int) or width < 1:
            raise ValueError("stroke_width must be an integer greater than or equal to 1.")
        self._stroke_width = width
    
    @property
    def texture(self) -> pygame.Surface|None:
        """
        Returns the current texture used for filling shapes.
        
        Returns:
            pygame.Surface: The current texture surface, or None if no texture is set.
        """
        return self._texture
    @texture.setter
    def texture(self, texture:pygame.Surface|None) -> None:
        """
        Sets the texture for filling shapes.
        
        Args:
            texture (pygame.Surface | None): The texture surface to set, or None to remove the texture.
        
        Raises:
            TypeError: If the texture is not a pygame.Surface or None.
        """
        if texture is not None and not isinstance(texture, pygame.Surface):
            raise TypeError("texture must be a pygame.Surface or None.")
        self._gradient = None  # Reset gradient if texture is set
        self._texture = texture
    
    @property
    def gradient(self) -> tuple[Color, Color]|None:
        """
        Returns the current gradient used for filling shapes.
        
        Returns:
            tuple[Color, Color]: A tuple of two Color instances representing the gradient colors, or None if no gradient is set.
        """
        return self._gradient
    @gradient.setter
    def gradient(self, colors:tuple[Color, Color]|None) -> None:
        """
        Sets the gradient for filling shapes.
        
        Args:
            colors (tuple[Color, Color] | None): A tuple of two Color instances representing the gradient colors, or None to remove the gradient.
        
        Raises:
            TypeError: If colors is not a tuple of two Color instances or None.
        """
        if colors is not None:
            if not isinstance(colors, tuple) or len(colors) != 2:
                raise TypeError("gradient must be a tuple of two Color instances or None.")
            if not all(isinstance(c, Color) for c in colors):
                raise TypeError("Both elements of the gradient must be Color instances.")
        self._texture = None
        self._gradient = colors
    
    
    @property
    def curve_detail(self) -> int:
        """
        Returns the current detail level for bezier curves.
        
        Returns:
            int: The current detail level for bezier curves.
        """
        return self._curve_detail
    @curve_detail.setter
    def curve_detail(self, detail:int) -> None:
        """
        Sets the detail level for bezier curves.
        
        Args:
            detail (int): The detail level to set for bezier curves.
        
        Raises:
            ValueError: If the detail is less than 1.
        """
        if not isinstance(detail, int) or detail < 1:
            raise ValueError("curve_detail must be an integer greater than or equal to 1.")
        self._curve_detail = detail
    
    # Pixel Access
    def set_pixel(self, x:int, y:int, color:Color) -> None:
        """
        Sets the pixel at (x, y) to the specified color.
        
        Args:
            x (int): The x coordinate of the pixel.
            y (int): The y coordinate of the pixel.
            color (Color): The color to set the pixel to.
        """
        self.set_at((x, y), color.to_tuple())
    
    def get_pixel(self, x:int, y:int) -> Color:
        """
        Gets the color of the pixel at (x, y).
        
        Args:
            x (int): The x coordinate of the pixel.
            y (int): The y coordinate of the pixel.
        
        Returns:
            Color: The color of the pixel at (x, y).
        """
        return Color(*self.get_at((x, y)))
    
    
    # Colors
    def no_fill(self) -> None:
        """
        Disables the fill color for shapes.
        This means shapes will not be filled with any color when drawn.
        """
        self._fill_color = Color(0, 0, 0, 0)
    
    def no_stroke(self) -> None:
        """
        Disables the stroke color for shapes.
        This means shapes will not have an outline when drawn.
        """
        self._stroke_color = Color(0, 0, 0, 0)
        self._stroke_width = 0
    
    def no_texture(self) -> None:
        """
        Disables the texture for shapes.
        This means shapes will not be filled with any texture when drawn.
        """
        self._texture = None
    
    def no_gradient(self) -> None:
        """
        Disables the gradient for shapes.
        This means shapes will not be filled with any gradient when drawn.
        """
        self._gradient = None
    
    
    # Drawing Methods
    def background(self, color:Color) -> None:
        """
        Fills the entire surface with the specified background color.
        
        Args:
            color (Color): The color to fill the surface with.
        """
        overlay = pygame.Surface(self.get_size(), pygame.SRCALPHA)
        overlay.fill(color.to_tuple())
        self.blit(overlay, (0, 0))
    
    
    def path(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a path defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the path.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Draw the path outline
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def shape(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a shape defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Fill the shape with the current fill color
        self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the shape
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    
    def point(self, x:float, y:float) -> None:
        """
        Draws a point at the specified position.
        
        Args:
            x (float): The x coordinate where the point should be drawn.
            y (float): The y coordinate where the point should be drawn.
        """
        
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("x and y must be numbers.")
        
        # Transform the point using the last transformation matrix
        transformed_point = self.last_transform @ Vector2D(x, y)
        
        # Draw the point as a small circle
        self._shape_outline(
            [transformed_point, transformed_point],
            self._stroke_color,
            self._stroke_width
        )
    
    def line(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws a line from (x1, y1) to (x2, y2).
        
        Args:
            x1 (float): The x coordinate of the start point.
            y1 (float): The y coordinate of the start point.
            x2 (float): The x coordinate of the end point.
            y2 (float): The y coordinate of the end point.
        """
        
        if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
            raise TypeError("All coordinates must be numbers.")
        
        # Transform points using the last transformation matrix
        start_point = self.last_transform @ Vector2D(x1, y1)
        end_point = self.last_transform @ Vector2D(x2, y2)
        
        # Draw the line outline
        self._shape_outline([start_point, end_point], self._stroke_color, self._stroke_width)
    
    def bezier(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a Bezier curve defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the Bezier curve.
            steps (int, optional): The number of steps to use for drawing the curve. Defaults to 100.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        if len(points) < 2:
            raise ValueError("At least two points are required to draw a Bezier curve.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Calculate Bezier curve points
        bezier_points = []
        for t in np.linspace(0, 1, self._curve_detail):
            ps = transformed_points
            for i, p in enumerate(transformed_points):
                for j, q in enumerate(transformed_points[:len(transformed_points) - 1 - i]):
                    ps[i] = ps[i] * (1 - t) + ps[i + 1] * t
            bezier_points.append(ps[0])
        
        # Draw the Bezier curve outline
        self._shape_outline(bezier_points, self._stroke_color, self._stroke_width)
    
    def cubic_bezier(self, x1:float, y1:float, x2:float, y2:float, x3:float, y3:float, x4:float, y4:float) -> None:
        """
        Draws a cubic Bezier curve defined by four control points.
        
        Args:
            x1 (float): The x coordinate of the first control point.
            y1 (float): The y coordinate of the first control point.
            x2 (float): The x coordinate of the second control point.
            y2 (float): The y coordinate of the second control point.
            x3 (float): The x coordinate of the third control point.
            y3 (float): The y coordinate of the third control point.
            x4 (float): The x coordinate of the fourth control point.
            y4 (float): The y coordinate of the fourth control point.
        """
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(x1, y1),
            self.last_transform @ Vector2D(x2, y2),
            self.last_transform @ Vector2D(x3, y3),
            self.last_transform @ Vector2D(x4, y4)
        ]
        
        # Calculate cubic Bezier curve points
        bezier_points = []
        for t in np.linspace(0, 1, self._curve_detail):
            u = 1 - t
            point = u**3 * transformed_points[0] + \
                    3*u**2*t * transformed_points[1] + \
                    3*u*t**2 * transformed_points[2] + \
                    t**3 * transformed_points[3]
            bezier_points.append(point)
        
        # Draw the cubic Bezier curve outline
        self._shape_outline(bezier_points, self._stroke_color, self._stroke_width)
    
    def triangle(self, x1:float, y1:float, x2:float, y2:float, x3:float, y3:float) -> None:
        """
        Draws a triangle defined by three points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
            x3 (float): The x coordinate of the third point.
            y3 (float): The y coordinate of the third point.
        """
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(x1, y1),
            self.last_transform @ Vector2D(x2, y2),
            self.last_transform @ Vector2D(x3, y3)
        ]
        
        # Fill the triangle with the current fill color
        self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the triangle
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def rectangle(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws a rectangle defined by two points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
        """
        
        # Convert coordinates based on the current rectangle mode
        coords = self._coordinates(self._rect_mode, x1, y1, x2, y2)
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(coords[0], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1] + coords[3]),
            self.last_transform @ Vector2D(coords[0], coords[1] + coords[3])
        ]
        
        # Fill the rectangle with the current fill color
        if self.fill_color.a != 0:
            self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the rectangle
        if self.stroke_color.a != 0 and self.stroke_width > 0:
            self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def ellipse(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws an ellipse defined by two points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
        """
        
        # Convert coordinates based on the current ellipse mode
        left, top, width, height = self._coordinates(self._ellipse_mode, x1, y1, x2, y2)
        cx, cy = left + width * 0.5, top + height * 0.5
        
        ellipse_points = [self.last_transform @ Vector2D(
            cx + width * 0.5 * np.sin(2 * np.pi * i / self.curve_detail),
            cy + height * 0.5 * np.cos(2 * np.pi * i / self.curve_detail)
        ) for i in range(self.curve_detail)]
            
        # Fill the ellipse with the current fill color
        self._shape_fill(ellipse_points, self._fill_color)
        
        # Draw the outline of the ellipse
        self._shape_outline(ellipse_points, self._stroke_color, self._stroke_width)
    
    def image(self, image:pygame.Surface, x:float, y:float, w: float = None, h: float = None, smooth: bool = True) -> None:
        """
        Draws an image on the surface at the specified position.
        
        Args:
            image (pygame.Surface): The image to draw.
            x (float): The x coordinate where the image should be drawn.
            y (float): The y coordinate where the image should be drawn.
        """
        
        if not isinstance(image, pygame.Surface):
            raise TypeError("image must be a pygame.Surface instance.")
        
        if not w or not h:
            w, h = image.get_size()
        
        # Convert coordinates based on the current image mode
        coords = self._coordinates(self._image_mode, x, y, w, h)
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(coords[0], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1] + coords[3]),
            self.last_transform @ Vector2D(coords[0], coords[1] + coords[3])
        ]
        
        # Draw the image at the transformed position
        self._shape_texture(transformed_points, image, uvs=[
            Vector2D(0, 0),
            Vector2D(1, 0),
            Vector2D(1, 1),
            Vector2D(0, 1)
        ], smooth=smooth)
    
    def arc (self, x:float, y:float, radius:float, start_angle:float, end_angle:float) -> None:
        """
        Draws an arc defined by a center point, radius, and start and end angles.
        
        Args:
            x (float): The x coordinate of the center of the arc.
            y (float): The y coordinate of the center of the arc.
            radius (float): The radius of the arc.
            start_angle (float): The starting angle of the arc in radians.
            end_angle (float): The ending angle of the arc in radians.
        """
        
        if not all(isinstance(coord, (int, float)) for coord in [x, y, radius, start_angle, end_angle]):
            raise TypeError("x, y, radius, start_angle, and end_angle must be numbers.")
        
        # Transform the center point using the last transformation matrix
        center = self.last_transform @ Vector2D(x, y)
        
        # Calculate points for the arc
        arc_points = []
        for angle in np.linspace(start_angle, end_angle, self._curve_detail):
            point = center + Vector2D(radius * np.cos(angle), radius * np.sin(angle))
            arc_points.append(point)
        
        # Draw the arc outline
        self._shape_outline(arc_points, self._stroke_color, self._stroke_width)
    
    
    # Text Methods
    def text(self, text:str, x:float, y:float) -> None:
        """
        Draws text on the surface at the specified position.
        
        Args:
            text (str): The text to draw.
            x (float): The x coordinate where the text should be drawn.
            y (float): The y coordinate where the text should be drawn.
        """
        
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        
        lines = text.split('\n')
        line_offset = self._text_font.get_height() + self._text_leading
        for i, line in enumerate(lines):
            # Create a text surface
            text_surface = self._text_font.render(line, True, self._fill_color.to_tuple())
            
            # Calculate position based on alignment
            left = x - self._text_align_x * text_surface.get_width()
            top = y - self._text_align_y * text_surface.get_height()
            position = (left, top)
            
            transformed_points = [self.last_transform @ vector for vector in [
                Vector2D(position[0], position[1] + line_offset * i),
                Vector2D(position[0] + text_surface.get_width(), position[1] + line_offset * i),
                Vector2D(position[0] + text_surface.get_width(), position[1] + text_surface.get_height() + line_offset * i),
                Vector2D(position[0], position[1] + text_surface.get_height() + line_offset * i)
            ]]
            
            # Draw the text surface at the transformed position
            self._shape_texture(transformed_points, text_surface, uvs=[
                Vector2D(0, 0),
                Vector2D(1, 0),
                Vector2D(1, 1),
                Vector2D(0, 1)
            ])
    
    @staticmethod
    def list_fonts() -> list[str]:
        """
        Lists all available fonts in Pygame.
        
        Returns:
            list[str]: A list of font names available in Pygame.
        """
        return pygame.font.get_fonts()
    
    def text_width(self, text: str) -> int:
        return self._text_font.size(text)[0]
    
    # Transformations
    def push_matrix(self) -> None:
        """
        Saves the current transformation matrix onto the stack.
        This allows for nested transformations.
        """
        self._transforms.append(self._transforms[-1].copy())
        return self._MatrixContext(self)
    
    def pop_matrix(self) -> None:
        """
        Restores the last transformation matrix from the stack.
        This undoes the last push_matrix call.
        
        Raises:
            IndexError: If there are no matrices to pop.
        """
        if len(self._transforms) <= 1:
            raise IndexError("No transformation matrix to pop.")
        self._transforms.pop()
    
    def reset_matrix(self) -> None:
        """
        Resets the transformation matrix to the identity matrix.
        This clears all transformations applied so far.
        """
        self._transforms = [Matrix2D.identity()]
    
    def apply_matrix(self, matrix:Matrix2D) -> None:
        """
        Applies a transformation matrix to the current transformation stack.
        
        Args:
            matrix (Matrix2D): The transformation matrix to apply.
        
        Raises:
            TypeError: If the matrix is not an instance of Matrix2D.
        """
        if not isinstance(matrix, Matrix2D):
            raise TypeError("matrix must be an instance of Matrix2D.")
        self._transforms[-1] = self.last_transform @ matrix
    
    def translate(self, x:float, y:float) -> None:
        """
        Translates the current transformation matrix by (x, y).
        
        Args:
            x (float): The x translation amount.
            y (float): The y translation amount.
        """
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("x and y must be numbers.")
        self.last_transform = self.last_transform.translate(x, y)
    
    def rotate(self, angle:float) -> None:
        """
        Rotates the current transformation matrix by the specified angle in radians.
        
        Args:
            angle (float): The angle in radians to rotate the matrix.
        
        Raises:
            TypeError: If the angle is not a number.
        """
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be a number.")
        self.last_transform = self.last_transform.rotate(angle)
    
    def scale(self, sx:float, sy:float) -> None:
        """
        Scales the current transformation matrix by (sx, sy).
        
        Args:
            sx (float): The x scale factor.
            sy (float): The y scale factor.
        
        Raises:
            TypeError: If sx or sy is not a number.
        """
        if not isinstance(sx, (int, float)) or not isinstance(sy, (int, float)):
            raise TypeError("sx and sy must be numbers.")
        self.last_transform = self.last_transform.scale(sx, sy)
    
    class _MatrixContext:
        def __init__(self, graphics:Graphics):
            """
            Context manager for applying transformations to the Graphics object.
            
            Args:
                graphics (Graphics): The Graphics object to apply transformations to.
            """
            self.graphics = graphics
        def __enter__(self):
            """
            Enters the context manager, saving the current transformation matrix.
            """
            self.graphics.push_matrix()
            return self.graphics
        def __exit__(self, exc_type, exc_value, traceback):
            """
            Exits the context manager, restoring the last transformation matrix.
            
            Args:
                exc_type: The exception type, if any.
                exc_value: The exception value, if any.
                traceback: The traceback object, if any.
            """
            self.graphics.pop_matrix()
    
    
    # Abstractions
    def _shape_outline(self, points:Iterable[Vector2D], color:Color=None, width:int=1) -> None:
        """
        Draws an outline of a shape defined by a list of points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
            color (Color, optional): The color of the outline. Defaults to the current stroke color.
            width (int, optional): The width of the outline. Defaults to 1.
        """
        
        if color is None:
            color = self._stroke_color
        
        if color.a == 255:
            # Fully opaque, draw directly
            pygame.draw.lines(self, color.to_tuple(), True, [p.to_tuple() for p in points], width)
        else:
            # Transparent outline, draw on temp surface
            temp_surf = pygame.Surface(self.get_size(), pygame.SRCALPHA)
            pygame.draw.lines(temp_surf, color.to_tuple(), True, [p.to_tuple() for p in points], width)
            self.blit(temp_surf, (0, 0))
    
    def _shape_fill(self, points:Iterable[Vector2D], color:Color=None) -> None:
        """
        Fills a shape defined by a list of points with the specified color.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
            color (Color, optional): The color to fill the shape with. Defaults to the current fill color.
        """
        
        if color is None:
            color = self._fill_color
        pygame.draw.polygon(self, color.to_tuple(), [p.to_tuple() for p in points])
    
    # FIXME: Nonsense
    def _shape_texture(self, points:Iterable[Vector2D], texture:pygame.Surface, uvs:Iterable[Vector2D]=None, smooth:bool=True) -> None:
        # Draw a textured polygon using the given points and UVs.
        if uvs is None or len(points) != len(uvs):
            raise ValueError("UVs must be provided and match the number of points.")

        # Convert points and uvs to lists of tuples
        dst_pts = [p.to_tuple() for p in points]
        src_w, src_h = texture.get_width(), texture.get_height()
        src_pts = [(uv.x * src_w, uv.y * src_h) for uv in uvs]

        # Create a mask surface for the polygon
        mask = pygame.Surface(self.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(mask, (255, 255, 255, 255), dst_pts)

        # Compute affine transform from src_pts to dst_pts (only works for quads/triangles)
        if len(points) == 4:
            # Use pygame.transform for quadrilaterals
            # Approximate by transforming the texture to fit the bounding rect of dst_pts
            min_x = min(p[0] for p in dst_pts)
            min_y = min(p[1] for p in dst_pts)
            max_x = max(p[0] for p in dst_pts)
            max_y = max(p[1] for p in dst_pts)
            dst_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
            src_rect = pygame.Rect(0, 0, src_w, src_h)
            # Scale texture to fit destination rect
            if smooth:
                tex_scaled = pygame.transform.smoothscale(texture, (dst_rect.width, dst_rect.height))
            else:
                tex_scaled = pygame.transform.scale(texture, (dst_rect.width, dst_rect.height))
            # Blit the scaled texture onto a temp surface
            temp = pygame.Surface(self.get_size(), pygame.SRCALPHA)
            temp.blit(tex_scaled, dst_rect.topleft)
            # Mask the polygon area
            temp.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.blit(temp, (0, 0))
        else:
            # For triangles or polygons, approximate by blitting the texture and masking
            temp = pygame.Surface(self.get_size(), pygame.SRCALPHA)
            temp.blit(texture, (0, 0))
            temp.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            self.blit(temp, (0, 0))
# TODO: Implement Shape drawing logic in Graphics

# === Animation ===
class Animation:
    # Static class with nested classes like Ease, Formula, Keyframe, Envelope and Timeline.
    @staticmethod
    def animate(t, ease, formula):
        """
        Animates a value based on time, easing function, and formula.
        Args:
            t (float): The time value between 0 and 1.
            ease (Ease): The easing function to apply.
            formula (Formula): The mathematical formula to apply.
        Returns:
            float: The animated value based on the easing function and formula.
        """
        return ease.value(formula.value(ease.time(t)), t)
class Ease:
    def __init__(self):
        """
        Static class for easing functions.
        """
        pass
    def time(self, t:float) -> float:
        """
        Argument for Formula.value().
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The eased value.
        """
        return t
    def value(self, t:float, v:float) -> float:
        """
        Final part of the equation.
        
        Args:
            t (float): The time value between 0 and 1.
            v (float): The value to ease.
        
        Returns:
            float: The eased value.
        """
        return v
Animation.Ease = Ease; del Ease
class In(Animation.Ease):
    """
    Easing function for ease-in.
    """
    def time(self, t:float) -> float:
        return t
    def value(self, t:float, v:float) -> float:
        return v
Animation.Ease.In = In; del In
class Out(Animation.Ease):
    """
    Easing function for ease-out.
    """
    def time(self, t:float) -> float:
        return 1 - t
    def value(self, t:float, v:float) -> float:
        return 1 - v
Animation.Ease.Out = Out; del Out
class InOut(Animation.Ease):
    """
    Easing function for ease-in-out.
    """
    def time(self, t:float) -> float:
        return 2 * t if t < 0.5 else 2 * (1 - t)
    def value(self, t:float, v:float) -> float:
        return v if t < 0.5 else 1 - v
Animation.Ease.InOut = InOut; del InOut
class OutIn(Animation.Ease):
    """
    Easing function for ease-out-in.
    """
    def time(self, t:float) -> float:
        return 1 - (2 * t if t < 0.5 else 2 * (1 - t))
    def value(self, t:float, v:float) -> float:
        return 1 - v if t < 0.5 else v
Animation.Ease.OutIn = OutIn; del OutIn
class Formula:
    def __init__(self):
        """
        Static class for mathematical formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a mathematical formula to the time value.
        Args:
            t (float): The time value between 0 and 1.
        Returns:
            float: The calculated value.
        """
        return t
Animation.Formula = Formula; del Formula
class Polynomial(Animation.Formula):
    """
    Polynomial formula for easing.
    """
    def __init__(self, degree:int=2):
        """
        Initializes the polynomial formula with a degree.
        
        Args:
            degree (int): The degree of the polynomial. Defaults to 2.
        """
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be an integer greater than or equal to 1.")
        self.degree = degree
    def value(self, t:float) -> float:
        """
        Applies the polynomial formula to the time value.
        Args:
            t (float): The time value between 0 and 1.
        Returns:
            float: The calculated value based on the polynomial formula.
        """
        if not isinstance(t, (int, float)):
            raise TypeError("t must be a number.")
        return t ** self.degree
Animation.Formula.Polynomial = Polynomial; del Polynomial
class Exponential(Animation.Formula):
    """
    Exponential formula for easing.
    """
    def __init__(self, base:float=2.0):
        """
        Initializes the exponential formula with a base.
        
        Args:
            base (float): The base of the exponential function. Defaults to 2.0.
        """
        if not isinstance(base, (int, float)) or base <= 0:
            raise ValueError("base must be a positive number.")
        self.base = base
    def value(self, t:float) -> float:
        """
        Applies the exponential formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the exponential formula.
        """
        if not isinstance(t, (int, float)):
            raise TypeError("t must be a number.")
        return self.base ** t
Animation.Formula.Exponential = Exponential; del Exponential
class Trigoniometric(Animation.Formula):
    def __init__(self):
        """
        Static class for trigonometric formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a trigonometric formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the trigonometric formula.
        """
        return np.sin(t * np.pi * 2)
Animation.Formula.Trigoniometric = Trigoniometric; del Trigoniometric
class Circular(Animation.Formula):
    def __init__(self):
        """
        Static class for circular formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a circular formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the circular formula.
        """
        return np.sqrt(1 - (t - 1) ** 2)
Animation.Formula.Circular = Circular; del Circular
class Keyframe:
    def __init__(self, time:float, value:float, ease:Animation.Ease=None):
        """
        Initializes a keyframe with a time, value, and optional easing function.
        
        Args:
            time (float): The time from the last keyframe.
            value (float): The value of the keyframe.
            ease (Animation.Ease, optional): The easing function to apply. Defaults to None.
        """
        if not isinstance(time, (int, float)):
            raise TypeError("time must be a number.")
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.time = time
        self.value = value
        self.ease = ease if ease is not None else Animation.Ease()
Animation.Keyframe = Keyframe; del Keyframe
class Envelope:
    def __init__(self, keyframes:list[Animation.Keyframe]=[], delay:float=0.0):
        """
        Initializes an envelope with a list of keyframes.
        
        Args:
            keyframes (list[Animation.Keyframe]): A list of keyframes defining the envelope.
        """
        if not isinstance(keyframes, list) or not all(isinstance(kf, Animation.Keyframe) for kf in keyframes):
            raise TypeError("keyframes must be a list of Animation.Keyframe instances.")
        self.keyframes = keyframes
        self.delay = delay
    def add_keyframe(self, keyframe:Animation.Keyframe) -> None:
        """
        Adds a keyframe to the envelope.
        Args:
            keyframe (Animation.Keyframe): The keyframe to add.
        Raises:
            TypeError: If keyframe is not an instance of Animation.Keyframe.
        """
        if not isinstance(keyframe, Animation.Keyframe):
            raise TypeError("keyframe must be an instance of Animation.Keyframe.")
        self.keyframes.append(keyframe)
    def remove_keyframe(self, keyframe:Animation.Keyframe) -> None:
        """
        Removes a keyframe from the envelope.
        Args:
            keyframe (Animation.Keyframe): The keyframe to remove.
        Raises:
            TypeError: If keyframe is not an instance of Animation.Keyframe.
        """
        if not isinstance(keyframe, Animation.Keyframe):
            raise TypeError("keyframe must be an instance of Animation.Keyframe.")
        self.keyframes.remove(keyframe)
    def __len__(self) -> int:
        """
        Returns the number of keyframes in the envelope.
        
        Returns:
            int: The number of keyframes.
        """
        return len(self.keyframes)
    def __getitem__(self, index:int) -> Animation.Keyframe:
        """
        Gets a keyframe by index.
        
        Args:
            index (int): The index of the keyframe to retrieve.
        
        Returns:
            Animation.Keyframe: The keyframe at the specified index.
        """
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        return self.keyframes[index]
    def __iter__(self):
        """
        Returns an iterator over the keyframes in the envelope.
        
        Returns:
            Iterator[Animation.Keyframe]: An iterator over the keyframes.
        """
        return iter(self.keyframes)
    def duration(self) -> float:
        """
        Calculates the total duration of the envelope based on the keyframes.
        
        Returns:
            float: The total duration of the envelope.
        """
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time
    def value_at(self, time:float) -> float:
        """
        Gets the value at a specific time based on the keyframes.
        
        Args:
            time (float): The time to get the value for.
        
        Returns:
            float: The value at the specified time.
        """
        if not isinstance(time, (int, float)):
            raise TypeError("time must be a number.")
        if not self.keyframes:
            return 0.0
        
        # Find the two keyframes surrounding the time
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].time <= time <= self.keyframes[i + 1].time:
                kf1 = self.keyframes[i]
                kf2 = self.keyframes[i + 1]
                t = (time - kf1.time) / (kf2.time - kf1.time)
                return Animation.Ease().value(t, kf1.value + (kf2.value - kf1.value) * t)
        
        # If time is after the last keyframe, return the last keyframe's value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        # If time is before the first keyframe, return the first keyframe's value
        return self.keyframes[0].value
    def copy(self) -> 'Animation.Envelope':
        """
        Creates a copy of the envelope.
        
        Returns:
            Animation.Envelope: A new envelope with the same keyframes and delay.
        """
        return Animation.Envelope(self.keyframes.copy(), self.delay)
Animation.Envelope = Envelope; del Envelope
class LiveValue:
    def __init__(self, value:float|int=0.0):
        """
        Initializes a live value that can be animated.
        
        Args:
            value (float|int, optional): The initial value. Defaults to 0.0.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.value = value
        self.base_value = value
        self.envelopes = []
    
    @property
    def current_value(self) -> float:
        """
        Gets the current value of the live value, modified by any active envelopes.
        
        Returns:
            float: The current value.
        """
        self.update()
        return self.value
    
    def update(self) -> None:
        """
        Updates the live value based on the active envelopes.
        This method should be called regularly to ensure the value is updated.
        """
        value = 0.0
        for envelope in self.envelopes:
            if time.time() - envelope.delay > envelope.duration():
                self.base_value += envelope.value_at(time.time())
            else:
                value += envelope.value_at(time.time() - envelope.delay)
        self.value = self.base_value + value
    def animate_now(self, envelope:Animation.Envelope) -> None:
        """
        Immediately applies an envelope to the live value.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        envelope.delay = time.time()
        self.envelopes.append(envelope.copy())
        self.update()
    def animate_after(self, envelope:Animation.Envelope, delay:float) -> None:
        """
        Applies an envelope to the live value after a specified delay.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
            delay (float): The delay in seconds before applying the envelope.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        if not isinstance(delay, (int, float)):
            raise TypeError("delay must be a number.")
        envelope.delay = time.time() + delay
        self.envelopes.append(envelope.copy())
    def animate_end(self, envelope:Animation.Envelope) -> None:
        """
        Ends the animation of an envelope.
        
        Args:
            envelope (Animation.Envelope): The envelope to end.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        # Calculate the time at which the all envelopes end
        end_time = time.time()
        for env in self.envelopes:
            if env.duration() + env.delay > end_time:
                end_time = env.duration() + env.delay
        envelope.delay = end_time
        self.envelopes.append(envelope.copy())
    def animate(self, envelope:Animation.Envelope) -> None:
        """
        Applies an envelope to the live value.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        self.envelopes.append(envelope.copy())
        self.update()
    
    def set_immediate_value(self, value:float|int) -> None:
        """
        Sets the live value immediately without animation.
        
        Args:
            value (float|int): The value to set.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.value = value
        self.base_value = value
    def set_ending_value(self, value:float|int) -> None:
        """
        Sets the live value to an ending value, clearing all envelopes.
        
        Args:
            value (float|int): The ending value to set.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        # Calculate final value
        final_value = self.get_ending_value()
        self.base_value = final_value - self.value + value
    def get_ending_value(self) -> float:
        """
        Gets the ending value of the live value, considering all envelopes.
        
        Returns:
            float: The ending value.
        """
        final_value = self.base_value
        for envelope in self.envelopes:
            final_value += envelope.value_at(envelope.duration())
        return final_value
Animation.LiveValue = LiveValue; del LiveValue


# === Yui ===
class Yui:
    def __init__(self, parent:Yui):
        # --- Transform ---
        self._x = 0
        self._y = 0
        self._r = 0
        self._sx = 1
        self._sy = 1
        self._ax = 0
        self._ay = 0
        
        # --- Matrix Cache & Flags ---
        self._local_matrix = Matrix2D.identity()
        self._world_matrix = Matrix2D.identity()
        self._local_inverted_matrix = Matrix2D.identity()
        self._world_inverted_matrix = Matrix2D.identity()
        self._needs_local_matrix_update = True
        self._needs_world_matrix_update = True

        # --- Graphics ---
        self._graphics = None
        self._needs_graphics_rebuild = True
        self._uses_graphics = False
        self._width = 1
        self._height = 1
        
        # --- Hierarchy ---
        self._parent = None
        self._children = []
        
        # --- Flags ---
        self._destroyed = False
        self._visible = True
        self._enabled = True

        # --- Debug ---
        self._draw_time_self = 0.0
        self._draw_time_subtree = 0.0

        self.set_parent(parent)
    
    def __getitem__(self, key:int|slice) -> 'Yui':
        """
        Gets a child Yui element by index or slice.
        
        Args:
            key (int|slice): The index or slice to access.
        
        Returns:
            Yui: The child Yui element(s).
        """
        if isinstance(key, int):
            return self._children[key]
        elif isinstance(key, slice):
            return self._children[key]
        else:
            raise TypeError("key must be an integer or slice.")
    def __setitem__(self, key:int, value:'Yui') -> None:
        raise NotImplementedError("Setting children directly is not supported. Use add_child() instead.")
    def __delitem__(self, key:int|slice) -> None:
        """
        Deletes a child Yui element by index.
        Args:
            key (int|slice): The index or slice to delete.
        Raises:
            TypeError: If key is not an integer or slice.
        """
        if isinstance(key, int):
            if 0 <= key < len(self._children):
                child = self._children[key]
                child.set_parent(None)
                del self._children[key]
            else:
                raise IndexError("Child index out of range.")
        elif isinstance(key, slice):
            for child in self._children[key]:
                child.set_parent(None)
            del self._children[key]
        else:
            raise TypeError("key must be an integer or slice.")
    def __len__(self) -> int:
        """
        Returns the number of child Yui elements.
        
        Returns:
            int: The number of children.
        """
        return len(self._children)
    
    @property
    def x(self) -> float:
        """
        Gets the x coordinate of this Yui element.
        
        Returns:
            float: The x coordinate.
        """
        return self._x
    @x.setter
    def x(self, value:float):
        """
        Sets the x coordinate of this Yui element.
        
        Args:
            value (float): The new x coordinate.
        """
        last_x = self.x
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number.")
        self._x = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(last_x, self.y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def y(self) -> float:
        """
        Gets the y coordinate of this Yui element.
        
        Returns:
            float: The y coordinate.
        """
        return self._y
    @y.setter
    def y(self, value:float):
        """
        Sets the y coordinate of this Yui element.
        
        Args:
            value (float): The new y coordinate.
        """
        last_y = self.y
        if not isinstance(value, (int, float)):
            raise TypeError("y must be a number.")
        self._y = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, last_y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def r(self) -> float:
        """
        Gets the rotation of this Yui element in radians.
        
        Returns:
            float: The rotation in radians.
        """
        return self._r
    @r.setter
    def r(self, value:float):
        """
        Sets the rotation of this Yui element in radians.
        
        Args:
            value (float): The new rotation in radians.
        """
        last_r = self.r
        if not isinstance(value, (int, float)):
            raise TypeError("r must be a number.")
        self._r = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, last_r, self.sx, self.sy, self.ax, self.ay)
    @property
    def sx(self) -> float:
        """
        Gets the x scale factor of this Yui element.
        
        Returns:
            float: The x scale factor.
        """
        return self._sx
    @sx.setter
    def sx(self, value:float):
        """
        Sets the x scale factor of this Yui element.
        
        Args:
            value (float): The new x scale factor.
        """
        last_sx = self.sx
        if not isinstance(value, (int, float)):
            raise TypeError("sx must be a number.")
        self._sx = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, last_sx, self.sy, self.ax, self.ay)
    @property
    def sy(self) -> float:
        """
        Gets the y scale factor of this Yui element.
        
        Returns:
            float: The y scale factor.
        """
        return self._sy
    @sy.setter
    def sy(self, value:float):
        """
        Sets the y scale factor of this Yui element.
        
        Args:
            value (float): The new y scale factor.
        """
        last_sy = self.sy
        if not isinstance(value, (int, float)):
            raise TypeError("sy must be a number.")
        self._sy = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, last_sy, self.ax, self.ay)
    @property
    def ax(self) -> float:
        """
        Gets the x anchor point of this Yui element.
        
        Returns:
            float: The x anchor point.
        """
        return self._ax
    @ax.setter
    def ax(self, value:float):
        """
        Sets the x anchor point of this Yui element.
        
        Args:
            value (float): The new x anchor point.
        """
        last_ax = self.ax
        if not isinstance(value, (int, float)):
            raise TypeError("ax must be a number.")
        self._ax = max(0, min(1, value))
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, last_ax, self.ay)
    @property
    def ay(self) -> float:
        """
        Gets the y anchor point of this Yui element.
        
        Returns:
            float: The y anchor point.
        """
        return self._ay
    @ay.setter
    def ay(self, value:float):
        """
        Sets the y anchor point of this Yui element.
        
        Args:
            value (float): The new y anchor point.
        """
        last_ay = self.ay
        if not isinstance(value, (int, float)):
            raise TypeError("ay must be a number.")
        self._ay = max(0, min(1, value))
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, self.ax, last_ay)
    @property
    def position(self) -> Vector2D:
        """
        Gets the position of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The position of the element.
        """
        return Vector2D(self._x, self._y)
    @position.setter
    def position(self, value:Vector2D|tuple):
        """
        Sets the position of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new position as a Vector2D or a tuple (x, y).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_position = self.position
        if isinstance(value, Vector2D):
            self._x, self._y = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._x, self._y = value
        else:
            raise TypeError("position must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(last_position.x, last_position.y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def rotation(self) -> float:
        """
        Gets the rotation of this Yui element in radians.
        
        Returns:
            float: The rotation in radians.
        """
        return self._r
    @rotation.setter
    def rotation(self, value:float):
        """
        Sets the rotation of this Yui element in radians.
        
        Args:
            value (float): The new rotation in radians.
        
        Raises:
            TypeError: If value is not a number.
        """
        last_rotation = self.rotation
        if not isinstance(value, (int, float)):
            raise TypeError("rotation must be a number.")
        self._r = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, last_rotation, self.sx, self.sy, self.ax, self.ay)
    @property
    def scale(self) -> Vector2D:
        """
        Gets the scale of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The scale of the element.
        """
        return Vector2D(self._sx, self._sy)
    @scale.setter
    def scale(self, value:Vector2D|tuple):
        """
        Sets the scale of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new scale as a Vector2D or a tuple (sx, sy).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_scale = self.scale
        if isinstance(value, Vector2D):
            self._sx, self._sy = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._sx, self._sy = value
        else:
            raise TypeError("scale must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, last_scale.x, last_scale.y, self.ax, self.ay)
    @property
    def anchor(self) -> Vector2D:
        """
        Gets the anchor point of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The anchor point of the element.
        """
        return Vector2D(self._ax, self._ay)
    @anchor.setter
    def anchor(self, value:Vector2D|tuple):
        """
        Sets the anchor point of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new anchor point as a Vector2D or a tuple (ax, ay).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_anchor = self.anchor
        if isinstance(value, Vector2D):
            self._ax, self._ay = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._ax, self._ay = value
        else:
            raise TypeError("anchor must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, last_anchor.x, last_anchor.y)
    
    # --- Matrix ---
    def _update_matrices(self):
        changed = False
        last_local_matrix = self._local_matrix
        last_local_matrix_inverted = self._local_inverted_matrix
        last_world_matrix = self._world_matrix
        last_world_inverted_matrix = self._world_inverted_matrix

        if self._needs_local_matrix_update:
            self._local_matrix = Matrix2D.identity()
            self._local_matrix = self._local_matrix.translate(self._x, self._y)
            self._local_matrix = self._local_matrix.rotate(self._r)
            self._local_matrix = self._local_matrix.translate(-self._ax * self._width * self._sx, -self._ay * self._height * self._sy)
            self._local_matrix = self._local_matrix.scale(self._sx, self._sy)
            self._local_inverted_matrix = self._local_matrix.invert()
            self._needs_local_matrix_update = False
            self._needs_world_matrix_update = True
            changed = True
        if self._needs_world_matrix_update or self._has_ancestor_requested_world_matrix_update():
            if self._parent is None:
                self._world_matrix = self._local_matrix
                self._world_inverted_matrix = self._local_inverted_matrix
            else:
                self._parent._update_matrices()
                self._world_matrix = self._parent.world_matrix @ self._local_matrix
                self._world_inverted_matrix = self._world_matrix.invert()
            changed = True
            self._needs_world_matrix_update = False
        if changed:
            self.on_matrix_updated(last_local_matrix, last_world_matrix)

    def _has_ancestor_requested_world_matrix_update(self) -> bool:
        """
        Checks if any ancestor has requested a world matrix update.
        
        Returns:
            bool: True if an ancestor has requested a world matrix update, False otherwise.
        """
        current = self._parent
        while current is not None:
            if current._needs_world_matrix_update:
                return True
            current = current._parent
        return False
    @property
    def local_matrix(self) -> Matrix2D:
        """
        Gets the local transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The local transformation matrix.
        """
        self._update_matrices()
        return self._local_matrix
    @property
    def world_matrix(self) -> Matrix2D:
        """
        Gets the world transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The world transformation matrix.
        """
        self._update_matrices()
        return self._world_matrix
    @property
    def local_inverted_matrix(self) -> Matrix2D:
        """
        Gets the inverted local transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The inverted local transformation matrix.
        """
        self._update_matrices()
        return self._local_inverted_matrix
    @property
    def world_inverted_matrix(self) -> Matrix2D:
        """
        Gets the inverted world transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The inverted world transformation matrix.
        """
        self._update_matrices()
        return self._world_inverted_matrix

    def to_local(self, point:Vector2D|tuple) -> Vector2D|tuple:
        """
        Converts a point from world coordinates to local coordinates.
        
        Args:
            point (Vector2D|tuple): The point in world coordinates.
        
        Returns:
            Vector2D|tuple: The point in local coordinates.
        """
        if isinstance(point, Vector2D):
            return self.world_inverted_matrix @ point
        elif isinstance(point, tuple) and len(point) == 2:
            tp = self.world_inverted_matrix @ Vector2D(*point)
            return (tp.x, tp.y)
        else:
            raise TypeError(f"point must be a Vector2D or a tuple of two numbers (not {type(point)}).")
    def to_world(self, point:Vector2D|tuple) -> Vector2D|tuple:
        """
        Converts a point from local coordinates to world coordinates.
        
        Args:
            point (Vector2D|tuple): The point in local coordinates.
        
        Returns:
            Vector2D|tuple: The point in world coordinates.
        """
        if isinstance(point, Vector2D):
            return self.world_matrix @ point
        elif isinstance(point, tuple) and len(point) == 2:
            tp = self.world_matrix @ Vector2D(*point)
            return (tp.x, tp.y)
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")
    def to_parent(self, point:Vector2D|tuple) -> Vector2D|tuple:
        """
        Converts a point from this element's local coordinates to its parent's local coordinates.

        Args:
            point (Vector2D|tuple): The point in this element's local coordinates.

        Returns:
            Vector2D|tuple: The point in the parent's local coordinates.
        """
        if self._parent is None:
            # No parent, so local == parent space
            if isinstance(point, Vector2D):
                return point
            elif isinstance(point, tuple) and len(point) == 2:
                return point
            else:
                raise TypeError("point must be a Vector2D or a tuple of two numbers.")
        if isinstance(point, Vector2D):
            return self._local_matrix @ point
        elif isinstance(point, tuple) and len(point) == 2:
            tp = self._local_matrix @ Vector2D(*point)
            return (tp.x, tp.y)
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")
    def is_in_local_bounds(self, point:Vector2D|tuple) -> bool:
        """
        Checks if a point is within the local bounds of this Yui element.
        
        Args:
            point (Vector2D|tuple): The point to check in local coordinates.
        
        Returns:
            bool: True if the point is within the local bounds, False otherwise.
        """
        if isinstance(point, Vector2D):
            x, y = point.x, point.y
        elif isinstance(point, tuple) and len(point) == 2:
            x, y = point
        else:
            raise TypeError(f"point must be a Vector2D or a tuple of two numbers, not {type(point)}.")
        
        bounds = self.local_bounds
        return (bounds[0] <= x <= bounds[0] + bounds[2]) and (bounds[1] <= y <= bounds[1] + bounds[3])
    @property
    def local_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the local bounds of this Yui element.
        
        Returns:
            tuple: A tuple (x, y, width, height) representing the local bounds.
        """
        return (0, 0, self.width, self.height)
    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the world bounds of this Yui element.
        
        Returns:
            tuple: A tuple (x, y, width, height) representing the world bounds.
        """
        if self._needs_world_matrix_update or self._has_ancestor_requested_world_matrix_update:
            self._update_matrices()
        l, t, r, b = self.local_bounds
        lt = self.world_matrix @ Vector2D(l, t)
        rt = self.world_matrix @ Vector2D(r, t)
        lb = self.world_matrix @ Vector2D(l, b)
        rb = self.world_matrix @ Vector2D(r, b)
        ar = [lt, rt, lb, rb]
        l = min([v.x for v in ar])
        t = min([v.y for v in ar])
        r = max([v.x for v in ar])
        b = max([v.y for v in ar])
        return (l, t, r, b)
    @property
    def local_subtree_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the axis-aligned bounding box of this element and all descendants in local coordinates.

        Returns:
            tuple: (min_x, min_y, max_x, max_y) in local coordinates.
        """
        # Start with this element's local bounding box
        l, t, r, b = self.local_bounds
        min_x, min_y = l, t
        max_x, max_y = r, b

        # Expand to include all children's subtree bounds (converted to our local space)
        for child in self._children:
            if not child.is_enabled: continue
            
            cl, ct, cr, cb = child.local_subtree_bounds
            # Convert each corner of the child's subtree bounds to our local space
            corners = [
                child.to_parent(Vector2D(cl, ct)),
                child.to_parent(Vector2D(cr, ct)),
                child.to_parent(Vector2D(cr, cb)),
                child.to_parent(Vector2D(cl, cb)),
            ]
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            min_x = min(min_x, *xs)
            min_y = min(min_y, *ys)
            max_x = max(max_x, *xs)
            max_y = max(max_y, *ys)
        
        return (min_x, min_y, max_x, max_y)
    @property
    def world_subtree_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the axis-aligned bounding box of this element and all descendants in world coordinates.

        Returns:
            tuple: (min_x, min_y, max_x, max_y) in world coordinates.
        """
        # Get local subtree bounds
        min_x, min_y, max_x, max_y = self.local_subtree_bounds
        # Convert all four corners to world coordinates
        corners = [
            self.to_world((min_x, min_y)),
            self.to_world((max_x, min_y)),
            self.to_world((max_x, max_y)),
            self.to_world((min_x, max_y)),
        ]
        xs = [c.x for c in corners]
        ys = [c.y for c in corners]
        return (min(xs), min(ys), max(xs), max(ys))
    
    # --- Hierarchy ---
    @property
    def parent(self) -> 'Yui':
        """
        Gets the parent of this Yui element.
        
        Returns:
            Yui: The parent element, or None if this is a root element.
        """
        return self._parent
    @property
    def children(self) -> list['Yui']:
        """
        Gets the children of this Yui element.
        
        Returns:
            list[Yui]: A list of child elements.
        """
        return self._children.copy()
    @property
    def child_count(self) -> int:
        """
        Gets the number of children of this Yui element.
        
        Returns:
            int: The number of child elements.
        """
        return len(self.children)
    @property
    def root(self) -> 'YuiRoot':
        """
        Gets the root element of this Yui element.
        
        Returns:
            YuiRoot: The root element.
        """
        current = self
        while current._parent is not None:
            current = current._parent
        return current
    @property
    def is_root(self) -> bool:
        """
        Checks if this Yui element is the root element.
        
        Returns:
            bool: True if this is the root element, False otherwise.
        """
        return isinstance(self, YuiRoot)
    @property
    def is_leaf(self) -> bool:
        """
        Checks if this Yui element is a leaf (has no children).
        
        Returns:
            bool: True if this element has no children, False otherwise.
        """
        return len(self._children) == 0
    @property
    def depth(self) -> int:
        """
        Gets the level of this Yui element in the hierarchy.
        
        Returns:
            int: The level of the element, where 0 is the root.
        """
        level = 0
        current = self._parent
        while current is not None:
            level += 1
            current = current._parent
        return level
    @property
    def height(self) -> int:
        """
        Gets the height of this Yui element.
        
        Returns:
            float: The height of the element, which is always 0 for a base Yui element.
        """
        return 0 if self.is_leaf else max(child.height for child in self._children) + 1
    @property
    def ancestors(self) -> list['Yui']:
        """
        Gets a list of all ancestor elements, starting from the immediate parent up to the root.

        Returns:
            list[Yui]: A list of ancestor elements, ordered from closest parent to root.
        """
        ancestors = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = current._parent
        return ancestors
    
    def is_ancestor_of(self, other: 'Yui') -> bool:
        if other is None: return False
        return other.is_descendant_of(self)
    def is_descendant_of(self, other: 'Yui') -> bool:
        return other in self.ancestors
    
    def set_parent(self, parent: 'Yui', index:int=None):
        if self._destroyed or isinstance(self, YuiRoot) or (parent is not None and parent.is_destroyed):
            return
        
        if parent is None: # Can't assign a null parent
            if self.parent:
                raise RuntimeError("Tried to assign a null parent on Yui init.")
            else:
                raise RuntimeError("Tried to assign a null parent on Yui parent change.")
        else:
            if parent.is_descendant_of(self): # Can only happen after init
                return
            
            if not self.parent: # Initializing
                index = max(0, min(parent.child_count, index if index else 0x7FFFFFFF))
                if not parent.can_child_be_added(self, index) or not self.can_parent_be_set(parent): # Can't be initialized with this parent, would default to None
                    raise RuntimeError("Can't assign this parent in Yui init.")
                self._parent = parent
                self._parent._children.insert(index, self)
                self.on_parent_set(None)
                self._parent.on_child_added(self, index)
            elif self._parent == parent:
                old_index = self._parent._children.index(self)
                new_index = max(0, min(self._parent.child_count - 1, index if index is not None else self._parent.child_count - 1))
                if not self._parent.can_child_be_moved(self, old_index, new_index): # No change if not allowed to move
                    return
                self._parent._children.remove(self)
                self._parent._children.insert(new_index, self)
                self._parent.on_child_moved(self, old_index, new_index)
            else: # Already initialized
                old_index = self._parent._children.index(self)
                new_index = max(0, min(parent.child_count, index))
                old_parent = self._parent
                if self.can_parent_be_set(parent) and self.can_parent_be_removed(self._parent) and self._parent.can_child_be_removed(self, old_index) and parent.can_child_be_added(self, new_index):
                    self._parent._children.remove(self)
                    self._parent = parent
                    self._parent._children.insert(index, self)
                    self.on_parent_removed(old_parent)
                    old_parent.on_child_removed(self, old_index)
                    self.on_parent_set(self._parent)
                    self._parent.on_child_added(self, new_index)
    def add_child(self, child:'Yui', index:int=None) -> None:
        """
        Adds a child Yui element to this Yui element.
        
        Args:
            child (Yui): The child element to add.
            index (int, optional): The index at which to add the child. Defaults to None, which adds at the end.
        
        Raises:
            TypeError: If child is not an instance of Yui.
            RuntimeError: If the child cannot be added due to hierarchy constraints.
        """
        if not isinstance(child, Yui):
            raise TypeError("child must be an instance of Yui.")
        if child.is_destroyed:
            raise RuntimeError("Cannot add a destroyed Yui element as a child.")
        if self._destroyed or isinstance(self, YuiRoot):
            raise RuntimeError("Cannot add children to a destroyed or root Yui element.")
        
        if index is None:
            index = len(self._children)
        else:
            index = max(0, min(len(self._children), index))
        
        if not self.can_child_be_added(child, index):
            raise RuntimeError("Cannot add this child to the parent Yui element.")
        
        child.set_parent(self, index)
    def add_children(self, children:list['Yui'], index:int=None) -> None:
        """
        Adds multiple child Yui elements to this Yui element.
        
        Args:
            children (list[Yui]): A list of child elements to add.
            index (int, optional): The index at which to add the children. Defaults to None, which adds at the end.
        
        Raises:
            TypeError: If any child is not an instance of Yui.
            RuntimeError: If any child cannot be added due to hierarchy constraints.
        """
        if not isinstance(children, list) or not all(isinstance(child, Yui) for child in children):
            raise TypeError("children must be a list of Yui instances.")
        if self._destroyed or isinstance(self, YuiRoot):
            raise RuntimeError("Cannot add children to a destroyed or root Yui element.")
        
        for child in children:
            if child.is_destroyed:
                raise RuntimeError("Cannot add a destroyed Yui element as a child.")
        
        if index is None:
            index = len(self._children)
        else:
            index = max(0, min(len(self._children), index))
        
        for child in children:
            self.add_child(child, index)

    # --- Graphics ---
    @property
    def graphics(self) -> 'Graphics':
        """
        Gets the graphics object associated with this Yui element.
        
        Returns:
            Graphics: The graphics object, or None if not set.
        """
        return self._graphics
    @property
    def uses_graphics(self) -> bool:
        """
        Checks if this Yui element uses graphics.
        
        Returns:
            bool: True if the element uses graphics, False otherwise.
        """
        return self._uses_graphics
    @uses_graphics.setter
    def uses_graphics(self, value:bool):
        """
        Sets whether this Yui element uses graphics.
        
        Args:
            value (bool): True to enable graphics, False to disable.
        """
        if not isinstance(value, bool):
            raise TypeError("uses_graphics must be a boolean.")
        if self._uses_graphics != value:
            self._uses_graphics = value
            if value:
                self._needs_graphics_rebuild = True
            else:
                self._graphics = None
                self._needs_graphics_rebuild = False
    
    @property
    def width(self) -> float:
        """
        Gets the width of this Yui element.
        
        Returns:
            float: The width of the element.
        """
        return self._width
    @width.setter
    def width(self, value:float):
        """
        Sets the width of this Yui element.
        
        Args:
            value (float): The new width.
        
        Raises:
            TypeError: If value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("width must be a number.")
        self._width = value
        self._needs_graphics_rebuild = True
    @property
    def height(self) -> float:
        """
        Gets the height of this Yui element.
        
        Returns:
            float: The height of the element.
        """
        return self._height
    @height.setter
    def height(self, value:float):
        """
        Sets the height of this Yui element.
        
        Args:
            value (float): The new height.
        
        Raises:
            TypeError: If value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("height must be a number.")
        self._height = value
        self._needs_graphics_rebuild = True
    @property
    def size(self) -> Vector2D:
        """
        Gets the size of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The size of the element.
        """
        return Vector2D(self._width, self._height)
    @size.setter
    def size(self, value:Vector2D|tuple):
        """
        Sets the size of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new size as a Vector2D or a tuple (width, height).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        if isinstance(value, Vector2D):
            self._width, self._height = value.x, value.y
            if self._uses_graphics:
                self._needs_graphics_rebuild = True
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._width, self._height = value
            if self._uses_graphics:
                self._needs_graphics_rebuild = True
        else:
            raise TypeError("size must be a Vector2D or a tuple of two numbers.")

    def _rebuild_graphics(self):
        """
        Rebuilds the graphics for this Yui element.
        This method should be overridden by subclasses to implement custom graphics rendering.
        """
        if not self._uses_graphics:
            return
        if self._destroyed:
            return
        if not self._needs_graphics_rebuild:
            return
        self._graphics = Graphics(self._width, self._height)
        self._needs_graphics_rebuild = False
    def draw(self, graphics:Graphics):
        """
        Draws this Yui element using the provided Graphics object.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        
        Raises:
            RuntimeError: If this Yui element does not use graphics.
        """
        if self._destroyed or not self._enabled or not self._visible:
            return
        
        if self.uses_graphics:
            debug_time_start = time.time()
            self._rebuild_graphics()
            self.on_draw(self._graphics)

            self._draw_time_self = time.time() - debug_time_start

            for child in self._children:
                child.draw(self._graphics)
            
            self.post_draw(graphics)
            
            graphics.push_matrix()
            graphics.apply_matrix(self.local_matrix)
            graphics.image_mode = 'corners'
            graphics.image(self._graphics, 0, 0)
            graphics.pop_matrix()
            
            self._draw_time_subtree = time.time() - debug_time_start
        else:
            debug_time_start = time.time()
            graphics.push_matrix()
            graphics.apply_matrix(self.local_matrix)
            self.on_draw(graphics)
        
            self._draw_time_self = time.time() - debug_time_start

            for child in self._children:
                child.draw(graphics)
            
            self.post_draw(graphics)
            graphics.pop_matrix()
                
            self._draw_time_subtree = time.time() - debug_time_start


    # --- Debug ---
    def print_tree(self):
        """
        Prints the hierarchy of this Yui element and its children.
        This is useful for debugging the structure of the Yui elements.
        """
        print(self.get_tree_string())
    def get_tree_string(self):
        """
        Prints the hierarchy of this Yui element and its children.
        This is useful for debugging the structure of the Yui elements.
        """
        indent = ' ' * (self.depth * 2)
        s = ""
        s += f"{indent}{self.__class__.__name__} (x={self.x}, y={self.y}, r={self.r}, sx={self.sx}, sy={self.sy}, width={self.width}, height={self.height})"
        for child in self.children:
            s += "\n" + child.get_tree_string()
        return s
    def draw_bounds(self, graphics:Graphics):
        """
        Draws the bounds of this Yui element for debugging purposes.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        """
        if self._destroyed:
            return
        
        name = self.__class__.__name__
        bounds = self.world_bounds
        # Hue based on depth
        color = Color(255, 255, 255, 255) # Normalize depth to a hue value

        graphics.no_fill()
        graphics.stroke_color = color # Stroke color based on depth
        graphics.stroke_width = 1
        graphics.rect_mode = 'corners' # Center mode for bounds
        graphics.rectangle(bounds[0], bounds[1], bounds[2], bounds[3])

        graphics.fill_color = color
        graphics.text_size = 12
        graphics.text_align = 0, 0 # Align top left
        graphics.text(name, bounds[0] + 2, bounds[1] + 2) # Draw name at top left of bounds
        
        for child in self._children:
            child.draw_bounds(graphics)

    # --- Flags ---
    @property
    def is_destroyed(self) -> bool:
        """
        Checks if this Yui element has been destroyed.
        
        Returns:
            bool: True if the element is destroyed, False otherwise.
        """
        return self._destroyed
    def destroy(self):
        if self.is_destroyed:
            return
        for child in self.children:
            child.destroy()
        self.on_destroyed()
        if self.parent is not None:
            self.parent.on_child_destroyed(self, self.parent._children.index(self))
            self.parent._children.remove(self)
        self._parent = None

    @property
    def is_enabled(self) -> bool:
        return self._enabled
    @is_enabled.setter
    def is_enabled(self, value: bool):
        self._enabled = value

    # --- Callbacks ---
    def on_transform_changed(self, last_x:float, last_y:float, last_r:float, last_sx:float, last_sy:float, last_ax:float, last_ay) -> None:
        """
        Callback for when the transformation of this Yui element changes.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def on_matrix_updated(self, local_matrix:Matrix2D, world_matrix:Matrix2D) -> None:
        """
        Callback for when the transformation matrices are updated.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def can_parent_be_set(self, parent: Yui) -> bool:
        """
        Checks if this Yui element can be set to the specified parent.
        This can be overriden by subclasses to implement custom logic.
        Args:
            parent (Yui): The parent element to check.
        Returns:
            bool: True if the parent can be set, False otherwise.
        """
        return True
    def can_parent_be_removed(self, parent: Yui) -> bool:
        """
        Checks if this Yui element can be removed from the specified parent.
        This can be overridden by subclasses to implement custom logic.
        Args:
            parent (Yui): The parent element to check.
        Returns:
            bool: True if the parent can be removed, False otherwise.
        """
        return True
    def can_child_be_added(self, child:Yui, index:int) -> bool:
        """
        Checks if a child Yui element can be added to this Yui element at the specified index.
        This can be overridden by subclasses to implement custom logic.
        
        Args:
            child (Yui): The child element to check.
            index (int): The index at which to add the child.
        
        Returns:
            bool: True if the child can be added, False otherwise.
        """
        return True
    def can_child_be_removed(self, child:Yui, index:int) -> bool:
        """
        Checks if a child Yui element can be removed from this Yui element at the specified index.
        This can be overridden by subclasses to implement custom logic.
        
        Args:
            child (Yui): The child element to check.
            index (int): The index of the child to remove.
        
        Returns:
            bool: True if the child can be removed, False otherwise.
        """
        return True
    def can_child_be_moved(self, child:Yui, index_from:int, index_to:int) -> bool:
        """
        Checks if a child Yui element can be removed from this Yui element at the specified index.
        This can be overridden by subclasses to implement custom logic.
        
        Args:
            child (Yui): The child element to check.
            index (int): The index of the child to move from.
            index (int): The index of the child to move to.
        
        Returns:
            bool: True if the child can be removed, False otherwise.
        """
        return True
    def on_parent_set(self, parent:Yui) -> None:
        """
        Callback for when the parent of this Yui element is set.
        This can be overridden by subclasses to perform custom actions.
        
        Args:
            parent (Yui): The new parent element.
        """
        pass
    def on_parent_removed(self, old_parent:Yui) -> None:
        """
        Callback for when the parent of this Yui element is removed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            old_parent (Yui): The old parent element that was removed.
        """
        pass
    def on_child_added(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is added.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was added.
            index (int): The index at which the child was added.
        """
        pass
    def on_child_removed(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is removed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was removed.
            index (int): The index at which the child was removed.
        """
        pass
    def on_child_moved(self, child:Yui, old_index:int, new_index:int) -> None:
        """
        Callback for when a child Yui element is moved.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was moved.
            old_index (int): The previous index of the child.
            new_index (int): The new index of the child.
        """
        pass
    def on_child_destroyed(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is destroyed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was destroyed.
            index (int): The index of the child that was destroyed.
        """
        pass
    def on_destroyed(self) -> None:
        """
        Callback for when this Yui element is destroyed.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def on_draw(self, graphics:Graphics) -> None:
        """
        Callback for when this Yui element is drawn.
        This can be overridden by subclasses to perform custom drawing actions.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        """
        pass
    def post_draw(self, graphics:Graphics) -> None:
        """
        Callback for after this Yui's children elements were drawn.
        This can be overridden by subclasses to perform custom drawing actions.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        """
        pass
    def is_interactable(self, point:Vector2D|tuple) -> bool:
        """
        Checks if this Yui element is interactable at the given point.
        May be overridden by subclasses to implement custom interaction logic.
        
        Args:
            point (Vector2D|tuple): The point to check in world coordinates.

        
        Returns:
            bool: True if the element is interactable at the point, False otherwise.
        """
        return True
    def on_enabled_changed(self):
        """
        Callback for when this Yui element's enabled state has changed.
        This can be overridden by subclasses to perform custom actions.
        """
        pass

class YuiRoot(Yui):
    def __init__(self, width:int=800, height:int=600, framerate:int=60, name:str='Yui Window', is_resizable:bool=False):
        # TODO: Implement root-specific initialization logic.
        """
        Initializes a root Yui element.
        The root element does not have a parent and is the top of the hierarchy.
        """
        super().__init__(parent=None)

        self._name = name
        self._framerate = framerate
        self._window = None  # Placeholder for the window object, to be set later
        self._window_graphics: Graphics = None  # Placeholder for the window graphics object, to be set later
        self._mouse = Mouse(self)
        self._keyboard = Keyboard(self)

        self.width = width
        self.height = height
        
        self._auto_draw_bounds = False
        self._is_resizable = is_resizable
        self._auto_background = None

    @property
    def mouse(self):
        return self._mouse
    @property
    def keyboard(self):
        return self._keyboard
    
    def yui_at_point(self, point:Vector2D, extends, exclude:list[Yui]=[], current:Yui=None) -> 'Yui':
        """
        Searches for a Yui element at the given point in world coordinates.
        
        Args:
            point (Vector2D): The point in world coordinates to search for a Yui element.
            extends (type): The type of Yui element to search for.
            exclude (list[Yui], optional): A list of Yui elements to exclude from the search. Defaults to an empty list.
            current (Yui, optional): The current Yui element being checked. Defaults to self.
        
        Returns:
            Yui: The first Yui element found at the point that matches the specified type, or None if no such element is found.
        """
        
        if current is None:
            current = self
        
        if current._destroyed or not current._enabled:
            return None

        local = current.to_local(point)
        
        # Check if a Yui has forced cutoff
        if current.uses_graphics and current is not YuiRoot:
            if not current.is_interactable(local):
                return None
        
        # Check if a child is interactable (from last to first)
        for child in reversed(current.children):
            pointed = self.yui_at_point(point, extends, exclude, child)
            if pointed is not None:
                return pointed
        
        # Check if the current Yui is the searched type
        if not isinstance(current, extends) and current not in exclude:
            return None
    
        if not current.is_in_local_bounds(local):
            return None
        
        if not current.is_interactable(local):
            return None
        
        return current
    def init(self):
        "Initlizes the pygame window and the Yui root element."
        
        pygame.init()
        pygame.font.init()
        self._window = pygame.display.set_mode((self._width, self._height))
        self._window_graphics = Graphics(self._window.get_width(), self._window.get_height())
        self._game_loop()
    def _game_loop(self):
        """
        The main game loop for the Yui root element.
        This method should be called to start the Yui application.
        """
        clock = pygame.time.Clock()
        
        while not self.is_destroyed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.destroy()
                    # Handle window destruction
                    pygame.quit()
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.width, self.height = event.w, event.h
                    self._window = pygame.display.set_mode((self._width, self._height), pygame.RESIZABLE)
                    self._window_graphics = Graphics(*self._window_graphics.get_size())
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._mouse.mouse_pressed(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._mouse.mouse_released(event)
                elif event.type == pygame.MOUSEMOTION:
                    self._mouse.mouse_moved(event)
                elif event.type == pygame.MOUSEWHEEL:
                    self._mouse.mouse_wheel(event)
                elif event.type == pygame.KEYDOWN:
                    self._keyboard.key_pressed(event)
                elif event.type == pygame.KEYUP:
                    self._keyboard.key_released(event)
                elif event.type == pygame.WINDOWENTER:
                    pass
                elif event.type == pygame.WINDOWLEAVE:
                    pass
                elif event.type == pygame.WINDOWFOCUSGAINED:
                    pass
                elif event.type == pygame.WINDOWFOCUSLOST:
                    pass
                elif event.type == pygame.WINDOWMINIMIZED:
                    pass
                elif event.type == pygame.WINDOWRESTORED:
                    pass
                elif event.type == pygame.DROPFILE:
                    pass
                elif event.type == pygame.DROPBEGIN:
                    pass
                elif event.type == pygame.DROPCOMPLETE:
                    pass
                

            # Handle UI-level resize
            if self.width != self._window.get_width() or self.height != self._window.get_height():
                # Update the window size if it has changed
                self._window = pygame.display.set_mode((self._width, self._height), pygame.RESIZABLE)
                self._window_graphics = Graphics(*self._window_graphics.get_size())
            
            # Handle drawing
            self._window_graphics.background(Color(0, 0, 0, 0))
            self.draw(self._window_graphics)
            self._do_debug(self._window_graphics)
            self._window.blit(self._window_graphics, (0, 0))
            
            pygame.display.flip()
            clock.tick(self._framerate)

    def draw(self, graphics):
        if isinstance(self._auto_background, Color):
            graphics.background(self._auto_background)
        super().draw(graphics)
        self._window_graphics.reset_matrix()
        if self._auto_draw_bounds:
            with graphics.push_matrix():
                self.draw_bounds(graphics)
    
    def _do_debug(self, graphics):
        if self._auto_draw_bounds:
            self.draw_bounds(graphics)
    @property
    def auto_draw_bounds(self) -> bool:
        return self._auto_draw_bounds
    @auto_draw_bounds.setter
    def auto_draw_bounds(self, value:bool):
        self._auto_draw_bounds = value
    @property
    def auto_background(self) -> Color:
        return self._auto_background
    @auto_background.setter
    def auto_background(self, value: Color):
        if isinstance(value, Color) or value is None:
            self._auto_background = value

# === IO ===
class Mouse:
    def __init__(self, root:YuiRoot):
        """
        Initializes a Mouse object that can be used to track mouse events.
        
        Args:
            root (YuiRoot): The root Yui element to which the mouse events will be related.
        """
        if root is None or not isinstance(root, YuiRoot):
            raise TypeError("root must be an instance of YuiRoot.")

        self._root = root
        self._pressed:'MouseListener' = None
        self._last_pressed:'MouseListener' = None
        self._start:'MouseEvent' = None
        self._last_pressed_left:'MouseEvent' = None
        self._last_pressed_right:'MouseEvent' = None
        self._last_pressed_middle:'MouseEvent' = None
        self._last_pressed_alt:'MouseEvent' = None
        self._last:'MouseEvent' = None
        self._current:'MouseEvent' = None
    
    @property
    def current(self):
        return self._current
    @property
    def last(self):
        return self._last
    @property
    def pressed(self):
        return self._pressed
    @property
    def root(self):
        return self._root
    
    def mouse_pressed(self, event:pygame.event.Event):
        """
        Handles mouse pressed events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if not isinstance(event, pygame.event.Event):
            raise TypeError("event must be a pygame.event.Event.")
        
        mouse_event = MouseEvent.PRESSED
        mouse_button = MouseEvent.mouse_button(event.button)
        last = self._current if self._current and self._current.any_button_down else None
        yui_event = MouseEvent(self, Vector2D(*event.pos), 0, mouse_event | mouse_button, last=last)
        
        pointed = self._root.yui_at_point(Vector2D(*event.pos), MouseListener)
        if pointed is None:
            return
        
        if self._start is None:
            self._start = yui_event
        
        button = yui_event.event & MouseEvent.BUTTON_MASK
        if button == MouseEvent.LEFT:
            self._last_pressed_left = yui_event
        elif button == MouseEvent.RIGHT:
            self._last_pressed_right = yui_event
        elif button == MouseEvent.MIDDLE:
            self._last_pressed_middle = yui_event
        elif button == MouseEvent.ALT:
            self._last_pressed_alt = yui_event
        
        if self._current is None:
            self._current = yui_event
        self._last = self._current
        self._current = yui_event

        self._pressed = pointed
        if self._pressed is not None:
            self._pressed.on_mouse_event(yui_event.to_local(self._pressed))
    def mouse_released(self, event:pygame.event.Event):
        """
        Handles mouse released events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if not isinstance(event, pygame.event.Event):
            raise TypeError("event must be a pygame.event.Event.")
        
        if self._pressed is None:
            return
        
        mouse_event = MouseEvent.RELEASED
        mouse_button = MouseEvent.mouse_button(event.button)
        last = self._current if self._current and self._current.any_button_down else None
        yui_event = MouseEvent(self, Vector2D(*event.pos), 0, mouse_event | mouse_button, last=last)
        if self._start is None:
            self._start = yui_event
        
        if self._current is None:
            self._current = yui_event
        
        self._last = self._current
        self._current = yui_event
        
        button = yui_event.event & MouseEvent.BUTTON_MASK
        if button == MouseEvent.LEFT:
            self._last_pressed_left = None
        elif button == MouseEvent.RIGHT:
            self._last_pressed_right = None
        elif button == MouseEvent.MIDDLE:
            self._last_pressed_middle = None
        elif button == MouseEvent.ALT:
            self._last_pressed_alt = None
        
        released = self._pressed
        if not self._current.any_button_down:
            self._pressed = None
        if released is not None:
            released.on_mouse_event(yui_event.to_local(released))

        if self._current.any_button_down:
            self._start = None
    def mouse_moved(self, event:pygame.event.Event):
        mouse_event = 0
        last = self._current if self._current and self._current.any_button_down else None
        yui_event = MouseEvent(self, Vector2D(*event.pos), 0, 0, last=last)
        if self._current is None:
            self._current = yui_event
        self._last = self._current
        self._current = yui_event
        
        # Move
        if self._pressed is None:
            pointed = self._root.yui_at_point(Vector2D(*event.pos), MouseListener)
            if pointed is not None:
                pointed.on_mouse_event(yui_event.to_local(pointed))
        # Drag
        else:
            self._pressed.on_mouse_event(yui_event.to_local(self._pressed))
    def mouse_wheel(self, event:pygame.event.Event):
        mouse_event = MouseEvent.WHEEL
        last = self._current if self._current and self._current.any_button_down else None
        scroll = event.y if hasattr(event, "y") else 0
        yui_event = MouseEvent(self, last.point if last else Vector2D(0, 0), scroll, mouse_event, last=last)
        
        if self._current is None:
            self._current = yui_event
        self._last = self._current
        self._current = yui_event

        if self._pressed is None:
            pointed = self._root.yui_at_point(yui_event.point, MouseListener)
            if pointed is not None:
                pointed.on_mouse_event(yui_event.to_local(pointed))
        else:
            self._pressed.on_mouse_event(yui_event.to_local(self._pressed))
    
    def pass_event(self, yui: MouseListener = None):
        if self._current is None:
            return
        self._pressed = None
        if yui is None:
            yui: MouseListener = self._root.yui_at_point(self._current.point, extends=MouseListener, exclude=[self])
        if yui is None:
            return
        yui.on_mouse_event(self._current.to_local(yui).pass_to(yui))
        self._pressed = yui
        
class MouseEvent:
    LEFT = 0x1
    RIGHT = 0x2
    MIDDLE = 0x4
    ALT = 0x8
    WHEEL = 0x10
    PRESSED = 0x20
    RELEASED = 0x40
    BUTTON_MASK = LEFT | RIGHT | MIDDLE | ALT
    TYPE_MASK = WHEEL | PRESSED | RELEASED

    def __init__(self, mouse:Mouse, point:Vector2D|tuple, scroll:int, event:int, timestamp:float=None, last:MouseEvent=None):
        """
        Initializes a mouse event.
        Args:
            mouse (Mouse): The mouse object that generated the event.
            point (Vector2D|tuple): The point where the event occurred.
            button (str): The button that was pressed or released.
            type (str): The type of the event (e.g., 'click', 'move').
            timestamp (float): The time when the event occurred.
        """
        self.mouse = mouse
        self.point = Vector2D(*point) if isinstance(point, tuple) else point
        self.scroll = scroll
        self.event = event if event else 0
        self.down = last.down if last else event & MouseEvent.BUTTON_MASK
        if last:
            if (self.event & MouseEvent.TYPE_MASK) == MouseEvent.PRESSED:
                self.down = last.down | (event & MouseEvent.BUTTON_MASK)
            if (self.event & MouseEvent.TYPE_MASK) == MouseEvent.RELEASED:
                self.down = last.down & ~(event & MouseEvent.BUTTON_MASK)
        self.timestamp = time.time() if timestamp is None else timestamp
        self.last = last
        self._passed = None
    
    def to_local(self, yui:MouseListener):
        transformed = yui.to_local(self.point)
        local = MouseEvent(self.mouse, transformed, self.scroll, self.event, self.timestamp)
        local.down = self.down
        local.last = self.last
        return local
    def to_world(self, yui:MouseListener):
        transformed = yui.to_world(self.point)
        local = MouseEvent(self.mouse, transformed, self.scroll, self.event, self.timestamp)
        local.down = self.down
        local.last = self.last
        return local
    def pass_to(self, yui: MouseListener):
        passed = MouseEvent(self.mouse, self.point, self.scroll, self.event, self.timestamp)
        passed.down = self.down
        passed.last = self.last
        passed._passed = yui
        return passed
    
    @staticmethod
    def mouse_button(button:int) -> int:
        """
        Gets the mouse button event type based on the button number.
        
        Args:
            button (int): The button number (1 for left, 2 for middle, 3 for right).
        
        Returns:
            int: The mouse button event type.
        """
        if button == 1:
            return MouseEvent.LEFT
        elif button == 2:
            return MouseEvent.MIDDLE
        elif button == 3:
            return MouseEvent.RIGHT
        else:
            return MouseEvent.ALT

    @property
    def is_passed(self) -> bool:
        return self._passed is not None
    @property
    def passed_from(self) -> MouseListener|None:
        return self._passed
    
    # --- Event Type Properties ---
    @property
    def is_pressed_event(self) -> bool:
        """
        Checks if the mouse event is a pressed event.
        
        Returns:
            bool: True if the event is a pressed event, False otherwise.
        """
        return (self.event & MouseEvent.PRESSED) != 0
    @property
    def is_released_event(self) -> bool:
        """
        Checks if the mouse event is a released event.
        
        Returns:
            bool: True if the event is a released event, False otherwise.
        """
        return (self.event & MouseEvent.RELEASED) != 0
    @property
    def is_wheel_event(self) -> bool:
        """
        Checks if the mouse event is a wheel event.
        
        Returns:
            bool: True if the event is a wheel event, False otherwise.
        """
        return (self.event & MouseEvent.WHEEL) != 0
    @property
    def is_move_event(self) -> bool:
        """
        Checks if the mouse event is a move event.
        
        Returns:
            bool: True if the event is a move event, False otherwise.
        """
        return (self.event & MouseEvent.TYPE_MASK) == 0 and self.scroll == 0

    # --- Event Button Properties ---
    @property
    def is_left_event(self) -> bool:
        """
        Checks if the mouse event is a left button event.
        
        Returns:
            bool: True if the event is a left button event, False otherwise.
        """
        return (self.event & MouseEvent.LEFT) != 0
    @property
    def is_right_event(self) -> bool:
        """
        Checks if the mouse event is a right button event.
        
        Returns:
            bool: True if the event is a right button event, False otherwise.
        """
        return (self.event & MouseEvent.RIGHT) != 0
    @property
    def is_middle_event(self) -> bool:
        """
        Checks if the mouse event is a middle button event.
        
        Returns:
            bool: True if the event is a middle button event, False otherwise.
        """
        return (self.event & MouseEvent.MIDDLE) != 0
    @property
    def is_alt_event(self) -> bool:
        """
        Checks if the mouse event is an alt button event.
        
        Returns:
            bool: True if the event is an alt button event, False otherwise.
        """
        return (self.event & MouseEvent.ALT) != 0

    # --- Buttons Down ---
    @property
    def is_left_down(self) -> bool:
        """
        Checks if the left mouse button is currently down.
        
        Returns:
            bool: True if the left button is down, False otherwise.
        """
        return (self.down & MouseEvent.LEFT) != 0
    @property
    def is_right_down(self) -> bool:
        """
        Checks if the right mouse button is currently down.
        
        Returns:
            bool: True if the right button is down, False otherwise.
        """
        return (self.down & MouseEvent.RIGHT) != 0
    @property
    def is_middle_down(self) -> bool:
        """
        Checks if the middle mouse button is currently down.
        Returns:
            bool: True if the middle button is down, False otherwise.
        """
        return (self.down & MouseEvent.MIDDLE) != 0
    @property
    def is_alt_down(self) -> bool:
        """
        Checks if the alt mouse button is currently down.
        Returns:
            bool: True if the alt button is down, False otherwise.
        """
        return (self.down & MouseEvent.ALT) != 0
    @property
    def any_button_down(self) -> bool:
        """
        Checks if any mouse button is currently down.
        
        Returns:
            bool: True if any button is down, False otherwise.
        """
        return self.down != 0

    # --- Misc Properties ---
    def dist_sq(self, point:Vector2D|tuple = Vector2D(0, 0)) -> float:
        """
        Calculates the squared distance from the mouse event point to another point.
        
        Args:
            point (Vector2D|tuple): The point to calculate the distance to.
        
        Returns:
            float: The squared distance from the mouse event point to the given point.
        """
        if isinstance(point, Vector2D):
            return self.point.distance_sq(point)
        elif isinstance(point, tuple) and len(point) == 2:
            return self.point.distance_sq(Vector2D(*point))
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")
    def dist(self, point:Vector2D|tuple = Vector2D(0, 0)) -> float:
        """
        Calculates the distance from the mouse event point to another point.
        
        Args:
            point (Vector2D|tuple): The point to calculate the distance to.
        
        Returns:
            float: The distance from the mouse event point to the given point.
        """
        if isinstance(point, (Vector2D, tuple)):
            return np.sqrt(self.dist_sq(point))
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")

    @property
    def is_invalid(self) -> bool:
        """
        Checks if the mouse event is invalid.
        
        Returns:
            bool: True if the event is invalid, False otherwise.
        """
        event_type = self.event & MouseEvent.TYPE_MASK
        event_button = self.event & MouseEvent.BUTTON_MASK

        # Multiple Types
        if (event_type & (event_type - 1)) != 0:
            return True

        # Mouse button specified on mouse wheel
        if event_type == MouseEvent.WHEEL and event_button != 0:
            return True
        
        # Scroll not specified on mouse wheel event
        if event_type == MouseEvent.WHEEL and self.scroll == 0:
            return True
        
        # Scroll specified on mouse button event or mouse button not specified on mouse button event
        if (event_type == MouseEvent.PRESSED or event_type == MouseEvent.RELEASED) and (self.scroll != 0 or event_button == 0):
            return True

        return False

class MouseListener(ABC):
    """
    Abstract base class for mouse listeners.
    Subclasses should implement the on_mouse_event method to handle mouse events.
    """
    @abstractmethod
    def on_mouse_event(self, event:MouseEvent):
        """
        Abstract method to handle mouse events.
        
        Args:
            event (MouseEvent): The mouse event to handle.
        """
        pass
    @abstractmethod
    def to_local(self, point:Vector2D) -> Vector2D:
        """
        All Yui implement this.
        
        Args:
            point (Vector2D): The point in world coordinates.
        
        Returns:
            Vector2D: The point in local coordinates.
        """
        pass

class Keyboard():
    def __init__(self, root:YuiRoot):
        self._root = root
        self._keys = []
        self._current:KeyboardListener = None
    
    @property
    def keys(self) -> list[KeyboardEvent]:
        """
        Gets the list of currently pressed keys.
        
        Returns:
            list[KeyboardEvent]: A list of KeyboardEvent objects representing the currently pressed keys.
        """
        return self._keys
    @property
    def listener(self) -> KeyboardListener:
        """
        Gets the current keyboard listener.
        
        Returns:
            KeyboardListener: The current keyboard listener handling keyboard events.
        """
        return self._current

    

    def key_pressed(self, event:pygame.event.Event):
        """
        Handles key pressed events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if not isinstance(event, pygame.event.Event):
            raise TypeError("event must be a pygame.event.Event.")
        
        key_event = KeyboardEvent(self, event)
        if any([key_event.is_key(k) for k in self._keys]):
            return
        self._keys.append(key_event)
        if self._current is not None:
            self._current.on_key_event(key_event)
    def key_released(self, event:pygame.event.Event):
        """
        Handles key released events.
        
        Args:
            event (pygame.event.Event): The pygame event to handle.
        """
        if not isinstance(event, pygame.event.Event):
            raise TypeError("event must be a pygame.event.Event.")
        
        key_event = KeyboardEvent(self, event)
        corresponding_key = None
        for i in range(len(self._keys)):
            if self._keys[i].is_key(key_event):
                corresponding_key = self._keys[i]
                break
        if corresponding_key is None:
            return
        self._keys.remove(corresponding_key)
        key_event = KeyboardEvent(self, event, last=corresponding_key)
        if self._current is not None:
            self._current.on_key_event(key_event)
    
    def start_keyboard(self, listener:KeyboardListener):
        """
        Starts listening for keyboard events with the specified listener.
        
        Args:
            listener (KeyboardListener): The listener to handle keyboard events.
        
        Raises:
            TypeError: If the listener is not an instance of KeyboardListener.
        """
        if not isinstance(listener, KeyboardListener):
            raise TypeError("listener must be an instance of KeyboardListener.")
        
        if self._current == listener:
            return
        if self._current is not None:
            self._current.on_keyboard_interupted(self, listener)
        self._current = listener
        self._current.on_keyboard_started(self)
    def end_keyboard(self):
        """
        Stops listening for keyboard events with the specified listener.
        
        Args:
            listener (KeyboardListener): The listener to stop handling keyboard events.
        
        Raises:
            TypeError: If the listener is not an instance of KeyboardListener.
        """
        if self._current is not None:
            self._current.on_keyboard_ended(self)
        self._current = None

class KeyboardEvent():
    def __init__(self, keyboard:Keyboard, event:pygame.event.Event, last:KeyboardEvent=None):
        """
        Initializes a keyboard event.
        
        Args:
            event (pygame.event.Event): The pygame event to convert.
        
        Raises:
            TypeError: If the event is not a pygame event.
        """

        self._keyboard = keyboard
        self._key = event.unicode
        self._key_code = event.key
        self._time = time.time()
        self._last:KeyboardEvent = last
    
    @property
    def keyboard(self):
        return self._keyboard
    @property
    def key(self) -> str:
        """
        Gets the key that was pressed.
        
        Returns:
            str: The key that was pressed.
        """
        return self._key
    @property
    def key_code(self) -> int:
        """
        Gets the key code of the pressed key.

        Returns:
            int: The key code of the pressed key.
        """
        return self._key_code
    @property
    def time(self) -> float:
        """
        Gets the time when the key was pressed.
        
        Returns:
            float: The time when the key was pressed.
        """
        return self._time
    @property
    def is_pressed(self) -> bool:
        """
        Checks if the key is currently pressed.
        
        Returns:
            bool: True if the key is pressed, False otherwise.
        """
        return self._last is not None
    @property
    def is_modifier(self) -> bool:
        """
        Checks if the key is a modifier key (e.g., Shift, Ctrl, Alt).
        
        Returns:
            bool: True if the key is a modifier key, False otherwise
        """
        return self._key_code in (pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_LALT, pygame.K_RALT, pygame.K_LMETA, pygame.K_RMETA)
    def is_key(self, key:str|int|KeyboardEvent) -> bool:
        """
        Checks if the key matches the specified key.
        
        Args:
            key (str|int): The key to check against.
            
        Returns:
            bool: True if the key matches, False otherwise.
        """
        if isinstance(key, KeyboardEvent):
            return self._key == key.key and self._key_code == key.key_code
        return self._key == key or self._key_code == key

class KeyboardListener(ABC):
    """
    Abstract base class for keyboard listeners.
    Subclasses should implement the on_key_event method to handle keyboard events.
    """
    @abstractmethod
    def on_key_event(self, event:KeyboardEvent):
        """
        Abstract method to handle keyboard events.
        
        Args:
            event (KeyboardEvent): The keyboard event to handle.
        """
        pass
    @abstractmethod
    def on_keyboard_started(self, keyboard:Keyboard) -> None:
        """
        Callback for when the keyboard listener is started.
        This can be overridden by subclasses to perform custom actions.
        
        Args:
            keyboard (Keyboard): The keyboard object that started the listener.
        """
        pass
    @abstractmethod
    def on_keyboard_ended(self, keyboard:Keyboard) -> None:
        """
        Callback for when the keyboard listener is stopped.
        This can be overridden by subclasses to perform custom actions.
        
        Args:
            keyboard (Keyboard): The keyboard object that stopped the listener.
        """
        pass
    @abstractmethod
    def on_keyboard_interupted(self, keyboard:Keyboard, cause:KeyboardListener) -> None:
        """
        Callback for when the keyboard listener is interrupted by another listener.
        This can be overridden by subclasses to perform custom actions.
        
        Args:
            keyboard (Keyboard): The keyboard object that was interrupted.
            cause (KeyboardListener): The listener that caused the interruption.
        """
        pass

# === Quick Implementations ===
class Button(Yui, MouseListener):
    """
    A simple button UI element that responds to mouse events.
    """
    def __init__(self, parent:Yui, label:str="Button", texture:pygame.Surface=None):
        super().__init__(parent)
        self.label = label
        self.texture = texture
    
    @property
    def is_hovered(self):
        return self.root.mouse.current and self.is_in_local_bounds(self.root.mouse.current.to_local(self).point)
    @property
    def is_pressed(self):
        return self.root.mouse.pressed == self

    def on_draw(self, graphics:Graphics):
        # Background color changes on hover/press
        if self.is_pressed:
            bg = Color(180, 180, 180)
        elif self.is_hovered:
            bg = Color(210, 210, 210)
        else:
            bg = Color(240, 240, 240)
        graphics.fill_color = bg
        graphics.stroke_color = Color(60, 60, 60)
        graphics.rect_mode = 'corner'
        graphics.rectangle(0, 0, self.width, self.height)

        # Draw texture if provided, else label
        if self.texture is not None:
            # Center the texture in the button
            tw, th = self.texture.get_width(), self.texture.get_height()
            tx = (self.width - tw) / 2
            ty = (self.height - th) / 2
            graphics.image(self.texture, tx, ty)
        else:
            graphics.fill_color = Color(0, 0, 0)
            graphics.text_size = 16
            graphics.text_align = (0, 0)
            tw = graphics._text_font.size(self.label)[0]
            th = graphics._text_font.size(self.label)[1]
            tx = (self.width - tw) / 2
            ty = (self.height - th) / 2
            graphics.text(self.label, tx, ty)

    def on_mouse_event(self, event:MouseEvent):
        if event.is_released_event and event.is_left_event:
            self.on_click()
    
    def on_click(self):
        pass

class Switch(Yui, MouseListener):
    """
    A simple rectangular switch UI element (toggle button).
    """
    
    class Radio:
        def __init__(self):
            self.switches = []
        
        def __len__(self):
            return len(self.switches)

        def __getitem__(self, idx):
            return self.switches[idx]

        def __iter__(self):
            return iter(self.switches)
            
        def add(self, switch: Switch):
            if switch not in self.switches:
                self.switches.append(switch)

        def remove(self, switch: Switch):
            if switch in self.switches:
                self.switches.remove(switch)
        
        def on_switched(self, switched: Switch):
            for s in self.switches:
                if s is not switched and s.checked:
                    s.checked = False
                    s.on_toggle(False)
    
    def __init__(self, parent:Yui, checked:bool=False, radio: 'Switch.Radio'|None = None):
        super().__init__(parent)
        self._checked = checked
        self._radio = radio
        if radio is not None: radio.add(self)
    
    @property
    def radio(self) -> Switch.Radio:
        return self._radio
    @property
    def checked(self) -> Switch.Radio:
        return self._checked
    @checked.setter
    def checked(self, value: bool):
        self._checked = value == True
    
    @property
    def is_hovered(self):
        return self.root.mouse.current and self.is_in_local_bounds(self.root.mouse.current.to_local(self).point)
    @property
    def is_pressed(self):
        return self.root.mouse.pressed == self

    def on_destroyed(self):
        self.radio.remove(self)
    
    def on_draw(self, graphics:Graphics):
        # Draw background
        if self.is_pressed:
            bg = Color(180, 220, 180) if self.checked else Color(220, 180, 180)
        elif self.is_hovered:
            bg = Color(200, 240, 200) if self.checked else Color(240, 200, 200)
        else:
            bg = Color(180, 255, 180) if self.checked else Color(255, 180, 180)
        graphics.fill_color = bg
        graphics.stroke_color = Color(60, 60, 60)
        graphics.rect_mode = 'corner'
        graphics.rectangle(0, 0, self.width, self.height)

        # Draw indicator rectangle
        margin = min(self.width, self.height) * 0.15
        knob_w = (self.width - 2 * margin) * 0.45
        knob_h = self.height - 2 * margin
        if self.checked:
            knob_x = self.width - margin - knob_w
        else:
            knob_x = margin
        knob_y = margin
        graphics.fill_color = Color(80, 80, 80)
        graphics.rectangle(knob_x, knob_y, knob_w, knob_h)

    def on_mouse_event(self, event:MouseEvent):
        if event.is_released_event and event.is_left_event:
            self.checked = not self.checked
            self.on_toggle(self.checked)
            if self.radio:
                self.radio.on_switched(self)
    
    def on_toggle(self, value:bool):
        pass

class Stack(Yui):
    def __init__(self, parent: Yui, is_vertical: bool = True):
        super().__init__(parent)
        
        self._is_vertical = is_vertical
        self._stack_align = 0.0
        self._stack_margin = 0.0
    
    @property
    def is_vertical(self) -> bool:
        return self._is_vertical
    
    @property
    def stack_align(self) -> float:
        return self._stack_align
    @stack_align.setter
    def stack_align(self, value:float):
        self._stack_align = max(0, min(1, value))
    @property
    def stack_margin(self) -> float:
        return self._stack_margin
    @stack_margin.setter
    def stack_margin(self, value:float):
        self._stack_margin = value
    
    def on_draw(self, graphics: Graphics):
        subtree_rectangles = []
        local_rectangles = []
        anchors = []
        
        enabled = [c for c in self._children if c.is_enabled]
        
        for i, child in enumerate(enabled):
            subtree = child.local_subtree_bounds
            points = [
                child.to_parent(Vector2D(subtree[0], subtree[1])),
                child.to_parent(Vector2D(subtree[2], subtree[1])),
                child.to_parent(Vector2D(subtree[0], subtree[3])),
                child.to_parent(Vector2D(subtree[2], subtree[3]))
            ]
            subtree = (
                min([p.x for p in points]),
                min([p.y for p in points]),
                max([p.x for p in points]),
                max([p.y for p in points]),
            )
            subtree_rectangles.append(subtree)
            
            local = child.local_bounds
            points = [
                child.to_parent(Vector2D(local[0], local[1])),
                child.to_parent(Vector2D(local[2], local[1])),
                child.to_parent(Vector2D(local[0], local[3])),
                child.to_parent(Vector2D(local[2], local[3]))
            ]
            local = (
                min([p.x for p in points]),
                min([p.y for p in points]),
                max([p.x for p in points]),
                max([p.y for p in points]),
            )
            local_rectangles.append(local)
            
            anchor = Vector2D(child.ax * child.width, child.ay * child.height)
            anchor = child.to_parent(anchor)
            anchors.append(anchor)
            
        width = max([r[2] - r[0] for r in subtree_rectangles]) if subtree_rectangles else 1
        height = max([r[3] - r[1] for r in subtree_rectangles]) if subtree_rectangles else 1
        total_width = sum([r[2] - r[0] for r in subtree_rectangles]) if subtree_rectangles else 1
        total_height = sum([r[3] - r[1] for r in subtree_rectangles]) if subtree_rectangles else 1
        
        coord = 0.0
        for i, (subtree, local, anchor, child) in enumerate(zip(subtree_rectangles, local_rectangles, anchors, enabled)):
            if self._is_vertical:
                offset_x = (width - (subtree[2] - subtree[0])) * self._stack_align
                child.x = offset_x + (anchor.x - subtree[0])
                child.y = coord + (anchor.y - subtree[1])
                coord += subtree[3] - subtree[1] + self._stack_margin
            else:
                offset_y = (height - (subtree[3] - subtree[1])) * self._stack_align
                child.y = offset_y + (anchor.y - subtree[1])
                child.x = coord + (anchor.x - subtree[0])
                coord += subtree[2] - subtree[0] + self._stack_margin
        
        self.width, self.height = total_width if not self.is_vertical else width, total_height if self.is_vertical else height

class TextField(Yui, MouseListener, KeyboardListener):
    def __init__(self, parent: Yui, is_editable: bool = True):
        super().__init__(parent)
        self._default_text = ""
        self._cursor = 0
        self._last_input_text = ""
        self._input_text = ""
        
        self._background_color = Color(23, 23, 23, 255)
        self._text_color = Color(0, 0, 0, 0)
        self._is_editable = is_editable
        self._cursor = 0
        
        self._clickthrough = None
    
    @property
    def is_editable(self) -> bool:
        return self._is_editable

    @is_editable.setter
    def is_editable(self, value: bool):
        self._is_editable = value

    @property
    def cursor(self) -> int:
        return self._cursor

    @cursor.setter
    def cursor(self, value: int):
        self._cursor = max(0, min(value, len(self._input_text)))

    @property
    def default_text(self) -> str:
        return self._default_text

    @default_text.setter
    def default_text(self, value: str):
        self._default_text = value

    @property
    def input_text(self) -> str:
        return self._input_text

    @input_text.setter
    def input_text(self, value: str):
        if not self._is_editable:
            raise ValueError("Cannot set input_text on a non-editable TextField.")

        if value != self._input_text:
            previous = self._input_text
            self._input_text = value
            self.cursor = self.cursor

            if self.is_focused:
                self.on_text_changed(previous)
            else:
                self.on_text_finalized(previous, interupted=False)
    
    @property
    def text_color(self) -> Color:
        return self._text_color

    @text_color.setter
    def text_color(self, value: Color):
        self._text_color = value

    @property
    def background_color(self) -> Color:
        return self.background_color

    @background_color.setter
    def background_color(self, value: Color):
        self.background_color = value

    @property
    def is_focused(self) -> bool:
        return not self.is_destroyed and self.root.keyboard.listener == self

    def on_draw(self, graphics: Graphics):
        if self._is_editable:
            graphics.fill_color = self._background_color
            graphics.rect_mode = 'corners'
            graphics.no_stroke()
            graphics.rectangle(0, 0, self.width, self.height)
        
        text_to_draw = self._input_text if self._input_text and len(self._input_text) > 0 else self._default_text

        graphics.fill_color = self._text_color
        graphics.text_size = int(max(self.height, 1))
        graphics.text_align_x = 0
        graphics.text_align_y = 0.5

        graphics.text(text_to_draw, 0, self.height * 0.5)

        if self.is_focused and self.is_editable:
            graphics.stroke_color = Color(self._text_color.r, self._text_color.g, self._text_color.b, int(np.sin(time.time() * np.pi * 2) * 127 + 128))
            graphics.stroke_width = 1
            caret_offset = graphics.text_width(self._input_text[:self._cursor])
            graphics.line(caret_offset, 0, caret_offset, self.height)
    
    def on_mouse_event(self, event: MouseEvent):
        if not event.is_pressed_event:
            return

        self.root.keyboard.start_keyboard(self)

    def on_keyboard_started(self, mouse: Mouse):
        if not self._is_editable:
            self.root.keyboard.end_keyboard()
        else:
            self._last_input_text = self._input_text
            self._clickthrough = self._new_clickthrough()
    
    def _new_clickthrough(self):
        class _ClickThrough(Yui, MouseListener):
            def __init__(self, text_field: TextField):
                super().__init__(text_field.root)
                
                self.text_field = text_field
            
            def on_draw(self, graphics):
                self.set_parent(self.root)
                self.x, self.y, self.width, self.height = 0, 0, self.root.width, self.root.height
            
            def on_mouse_event(self, event):
                if event.is_pressed_event:
                    text_field_event = event.to_world(self).to_local(self.text_field)
                    if self.text_field.is_in_local_bounds(text_field_event.point):
                        event.mouse.pass_event(yui=self.text_field)
                    else:
                        event.mouse.root.keyboard.end_keyboard()
                        event.mouse.pass_event()
                        
        return _ClickThrough(self)

    def on_keyboard_ended(self, keyboard: Keyboard):
        self.on_text_finalized(self._last_input_text, False)
        if self._clickthrough is not None:
            self._clickthrough.destroy()

    def on_keyboard_interupted(self, mouse: Mouse, cause: KeyboardListener):
        self.on_text_finalized(self._last_input_text, interupted=True)
        self._clickthrough.destroy()

    def on_key_event(self, event: KeyboardEvent):
        if not self.is_editable or not self.is_focused:
            return
        
        if not event.is_pressed:
            return 

        key = event.key_code
        ctrl = pygame.K_LCTRL in event.keyboard.keys or pygame.K_RCTRL in event.keyboard.keys

        # Cursor movement
        if key == pygame.K_LEFT:
            self.cursor -= 1
        elif key == pygame.K_RIGHT:
            self.cursor += 1
        elif key in (pygame.K_HOME, pygame.K_UP):
            self.cursor = 0
        elif key in (pygame.K_END, pygame.K_DOWN):
            self.cursor = len(self._input_text)

        # Confirm or cancel input
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self.root.keyboard.end_keyboard()

        # Delete backward
        elif key == pygame.K_BACKSPACE and self.cursor > 0:
            if ctrl:
                start = self._find_previous_word_start(self._input_text, self.cursor)
                self.input_text = self._input_text[:start] + self._input_text[self.cursor:]
                self.cursor = start
            else:
                adjust = self.cursor != len(self._input_text)
                self.input_text = self._input_text[:self.cursor - 1] + self._input_text[self.cursor:]
                if adjust:
                    self.cursor -= 1

        # Delete forward
        elif key == pygame.K_DELETE and self.cursor < len(self._input_text):
            if ctrl:
                end = self._find_next_word_end(self._input_text, self.cursor)
                self.input_text = self._input_text[:self.cursor] + self._input_text[end:]
            else:
                self.input_text = self._input_text[:self.cursor] + self._input_text[self.cursor + 1:]

        # Add character input
        elif len(event.key) == 1 and event.key.isprintable():
            self.input_text = self._input_text[:self.cursor] + event.key + self._input_text[self.cursor:]
            self.cursor += 1

    def _find_previous_word_start(self, text: str, index: int) -> int:
        index -= 1
        while index > 0 and text[index].isspace():
            index -= 1
        while index > 0 and not text[index - 1].isspace():
            index -= 1
        return max(index, 0)

    def _find_next_word_end(self, text: str, index: int) -> int:
        length = len(text)
        while index < length and text[index].isspace():
            index += 1
        while index < length and not text[index].isspace():
            index += 1
        return index

    # Callbacks to override or extend
    def on_text_changed(self, previous: str):
        pass

    def on_text_finalized(self, previous: str, interupted: bool):
        pass

class Slider(Yui, MouseListener):
    def __init__(self, parent: Yui, normalized_value: float = 0.5, range: tuple[float, float] = (0, 1), steps: int = 0):
        super().__init__(parent)
        self._normalized_value = max(0, min(1, normalized_value))
        self._range = list(range)          
        self._steps = 0 if steps < 2 else steps
    @property
    def dragging(self) -> bool:
        return self.root.mouse.pressed == self
    @property
    def normalized_value(self) -> float:
        return self._normalized_value
    @normalized_value.setter
    def normalized_value(self, value: float):    
        if self._steps >= 2:                
            value = round(value * (self._steps - 1)) / (self._steps - 1)            
        clamped = max(0, min(1, value))    
        if clamped == self._normalized_value:
            return        
        old = self._normalized_value
        self._normalized_value = clamped    
        self.on_value_changed(old)                
    @property                
    def steps(self) -> int:                
        return self._steps                
    @steps.setter                
    def steps(self, value: int):                
        self._steps = 0 if value < 2 else value  
    @property  
    def normalized_step_size(self) -> float:  
        return 1 / (self._steps - 1) if self._steps != 0 else 0
    @property  
    def step_size(self) -> float:  
        return self.normalized_step_size * (self.maximum - self.minimum)  
    @property
    def minimum(self) -> float:
        return self._range[0]
    @minimum.setter
    def minimum(self, value: float):
        self._range[0] = value
    @property
    def maximum(self) -> float:
        return self._range[1]
    @maximum.setter
    def maximum(self, value: float):
        self._range[1] = value
    @property
    def value(self) -> float:
        return self.minimum + self._normalized_value * (self.maximum - self.minimum)
    @value.setter
    def value(self, value: float):
        self.normalized_value = (value - self.minimum) / (self.maximum - self.minimum)
    def on_draw(self, graphics: Graphics):
        left = self.height / 2            
        right = self.width - left
        knob_size = self.height
        knob_x = left + self.normalized_value * (right - left)
        
        graphics.stroke_width = 1
        
        # Draw track
        graphics.fill_color = Color(200, 200, 200)
        graphics.stroke_color = Color(100, 100, 100)
        graphics.rect_mode = 'corner'
        graphics.rectangle(left, self.height / 2 - self.height / 4, right - left, self.height / 2)   
        
        # Draw markers
        for i in range(self._steps - 2): # Will not draw when steps is 0, excludes first and last
            graphics.stroke_color = Color(0, 0, 0, 255)
            x = left + (i + 1) * (right - left) / (self._steps - 1)
            graphics.line(x, self.height * 0.375, x, self.height * 0.625)

        # Draw knob
        graphics.fill_color = Color(80, 120, 220, 255) if self.dragging else Color(120, 160, 240, 255)
        graphics.stroke_color = Color(100, 100, 100, 255)
        graphics.rect_mode = 'center'
        graphics.rectangle(knob_x, self.height / 2, knob_size, knob_size)

    def on_mouse_event(self, event: MouseEvent):
        if event.any_button_down:
            self.normalized_value = (event.point.x - self.height / 2) / (self.width - self.height)
    def on_value_changed(self, old: float):
        pass

class TabView(Stack):
    class _TabStack(Stack):
        def __init__(self, parent: Yui):
            super().__init__(parent, is_vertical=False)
        def on_draw(self, graphics):
            for tab in self._children:
                label = tab.name
                graphics.text_size = int(self.height)
                tab.width = graphics.text_width(label) + 5
                tab.height = graphics.text_size
            super().on_draw(graphics)

        def can_parent_be_set(self, parent: Yui) -> bool:
            return isinstance(parent, TabView)
    
    class _Tab(Switch):
        def __init__(self, parent: Yui, container: 'TabView.Container', radio: Switch.Radio):
            super().__init__(parent, checked=False, radio=radio)
            self._container = container
            print(self.radio)
        @property
        def name(self) -> str:
            return self._container.name
        def on_destroyed(self, parent: Yui, index: int):
            if not self._container.is_destroyed:
                TabView._Tab(parent, parent._radio, self._container).set_parent(self.parent, index) # Cannot be destroyed
            super().on_destroyed()
        def on_toggle(self, value: bool):
            self._container.is_enabled = value
            # Ensure at least one tab is always checked
            siblings = [c for c in self.parent.children if isinstance(c, TabView._Tab)]
            if not any(tab.checked for tab in siblings):
                self.checked = True
                self._container.is_enabled = True
            
        def on_draw(self, graphics: Graphics):
            # Draw tab background based on checked state
            if self.checked:
                graphics.stroke_color = Color(180, 200, 255)
                graphics.stroke_width = 1
                graphics.line(0, self.height * 0.95, self.width, self.height * 0.95)
                graphics.fill_color = Color(180, 200, 255)
            else:
                graphics.fill_color = Color(191, 191, 191)
            graphics.text_align_x, graphics.text_align_y = 0.5, 0.5
            graphics.text_size = self.height
            graphics.text(self.name, self.width * 0.5, self.height * 0.5)
    
    class Container(Yui):
        def __init__(self, parent: Yui, tab_name: str):
            super().__init__(parent)
            self._tab_name = tab_name
        @property
        def name(self) -> str:
            return self._tab_name
    
    def __init__(self, parent: Yui):
        super().__init__(parent, is_vertical=True)
        self._radio = Switch.Radio()
        self._tab_bar = TabView._TabStack(self)

    @property
    def tab_stack_height(self) -> float:
        return self._children[0].height
    @tab_stack_height.setter
    def tab_stack_height(self, value: float):
        self._children[0].height = value
    
    @property
    def tab_stack_margin(self) -> float:
        return self._children[0].stack_margin
    @tab_stack_margin.setter
    def tab_stack_margin(self, value: float):
        self._children[0].stack_margin = value
    
    def on_draw(self, graphics):
        self._tab_bar.width = self.width
        
        super().on_draw(graphics)
    
    def can_child_be_added(self, child: Yui, index: int) -> bool:
        if isinstance(child, TabView._TabStack):
            for c in self:
                if isinstance(c, TabView._TabStack):
                    return False
            return True
        if isinstance(child, TabView.Container):
            return True
        return False
    
    def on_child_added(self, child: Yui, index: int):
        if isinstance(child, TabView._TabStack): return
        tab = TabView._Tab(self._tab_bar, child, self._radio)
        tab.set_parent(self._tab_bar, index - 1)
        if self.child_count == 2:
            tab.checked = True
        tab.on_toggle(tab.checked)
    
    def on_child_removed(self, child: Yui, index: int):
        self._tab_bar[index - 1].destroy()
    
    def can_child_be_moved(self, child: Yui, old: int, new: int) -> bool:
        return not isinstance(child, TabView._TabStack)
    
    def on_child_moved(self, child: Yui, old: int, new: int) -> bool:
        self._tab_bar[old - 1].set_parent(self._tab_bar, new - 1)


