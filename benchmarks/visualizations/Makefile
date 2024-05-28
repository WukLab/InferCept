make_pdf:
	jupyter nbconvert "Plot Over Model Sizes".ipynb --to=pdf --TemplateExporter.exclude_input=True --output "Plots Over Model Sizes".pdf

notebook:
	cp $(file).ipynb '$(title).ipynb'
	jupyter nbconvert '$(title).ipynb' --to=pdf --TemplateExporter.exclude_input=True --output=$(file)
	rm '$(title).ipynb'
