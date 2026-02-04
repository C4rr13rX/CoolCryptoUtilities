# Reprocess a fixed list of incomplete books, overwriting previous outputs.
# Uses the segmentation script with FORCE_REPROCESS to start from page 1.

$books = @(
  'Chem1Lower-full',
  'Chemistry1eOpenSTAX-full',
  'Chemistry2eOpenStax-full',
  'ChemistryAtomsFirst1eOpenSTAX-full',
  'ChemistryAtomsFirst2eOpenStax-full',
  'ChemistryForAlliedHealthSoult-full',
  'ChemistryForChangingTimesHillAndMcCreary-full',
  'ChemistryLabsTrufanAndBouhoutsosBrown-full',
  'CLUEChemistryLifeTheUniverseAndEverything-full',
  'ConceptDevelopmentStudiesInChemistryHutchinson-full',
  'CorporateGovernanceDeKluyver-full',
  'CorporateGovernanceFreyAndCruzCruz-full',
  'CropAdaptationAndImprovementForDroughtProneEnvironmentsKaneFoncKaAndDalton-full',
  'CropGeneticsSuzaAndLamkey-full'
)

foreach ($b in $books) {
  Write-Host "Reprocessing $b ..."
  $env:BOOK_FILTER = $b
  $env:MAX_BOOKS = 1
  $env:FORCE_REPROCESS = 'true'
  node tools/scripts/segment-textbook.mjs
}

Remove-Item Env:BOOK_FILTER, Env:MAX_BOOKS, Env:FORCE_REPROCESS -ErrorAction SilentlyContinue
