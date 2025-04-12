import vtk

# Read the .vtr file
reader = vtk.vtkXMLRectilinearGridReader()
reader.SetFileName("shape.vtr")
reader.Update()

# Map the data for visualization
mapper = vtk.vtkDataSetMapper()
mapper.SetInputConnection(reader.GetOutputPort())

# Create an actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Create a renderer, render window, and interactor
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Add actor to renderer
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)  # Background color

# Start interaction
render_window.Render()
interactor.Start()
