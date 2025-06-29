using MarkovRandomFields
using Documenter

DocMeta.setdocmeta!(MarkovRandomFields, :DocTestSetup, :(using MarkovRandomFields); recursive=true)

makedocs(;
    modules=[MarkovRandomFields],
    authors="stecrotti <sky_96@live.it>",
    sitename="MarkovRandomFields.jl",
    format=Documenter.HTML(;
        canonical="https://stecrotti.github.io/MarkovRandomFields.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stecrotti/MarkovRandomFields.jl",
    devbranch="main",
)
