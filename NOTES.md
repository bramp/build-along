
/Users/bramp/personal/lego-instructions/data/17101/17101_A_AutoBuilder.pdf
Saved: debug/all/17101_A_AutoBuilder.json (took 78.1s)

# This file is really slow
pants package src/build_a_long/pdf_extract:main 
scalene dist/src.build_a_long.pdf_extract/main.pex -- --output-dir debug/all  data/17101/17101_A_AutoBuilder.pdf

pants run src/build_a_long/pdf_extract:main -- --output-dir debug/all  data/17101/17101_A_AutoBuilder.pdf



pants run scripts:process-all-examples

# Falcon
pants run src/build_a_long/pdf_extract:main -- --debug-json --draw-elements --draw-unconsumed --debug-classification --output-dir debug data/75375/6509377.pdf --pages 1,5,10-17,45,70,149,172-176,180,126

# Christmas Tree
pants run src/build_a_long/pdf_extract:main -- --debug-json --draw-elements --draw-unconsumed --debug-classification --output-dir debug data/40573/6433200.pdf --pages 4,5,7,30,31 


pants run src/build_a_long/pdf_extract:main -- --output-dir debug --no-json --debug-classification data/75375/6509377.pdf --pages 126



--draw-

8073.17s. without clip
37.45s. normal 


 -- 

[ ] We have find_image_shadow_effects, but many Classifiers do something like this. Can we become consistent on how to do this task. I'm wondering if find_contained_effects should be in the default implementation of _get_additional_source_blocks. Please run `pants test ::` as you go. It is unacceptable to change the golden files.



[ ] The rules around how to write a Classifier are still a little ambigous. I'd like to document what is the best practie in score vs build. How it should work, what API is acceptable to use, etc. These rules should form the basis of documentation (or API changes) that humans and AI agents may use to write robust Classifiers.

[ ] We use TYPE_CHECKING in places, but I don't think I need this, we can use the __future__ annotation. Please remove it where it's not needed.

[ ] I would like to migrate more of the Classifiers over to the RuleBasedClassifier one.

[ ] Some of the RuleBasedClassifier validate text. It would be good to create a generic TextRule that can accept a function to do the validation.


[ ] I'm trying to reduce the number of unconsumed blocks to zero in the classifer. Please run `pants run src/build_a_long/pdf_extract:main -- --no-json --debug-unconsumed data/75375/6509377.pdf` and identify sources of unconsumed blocks, and let's fix them. The goal is to have all blocks consumed by LegoPage Elements

[ ] We have lots of rules in src/build_a_long/pdf_extract/classifier/rules, and the Scoring ones, typically return scores on a linear scale, a discrete scale, a boolean (filter) score, or some other scale. I'd like to standardise how the rules work, and perhaps abstract the scale from the rule itself (or make it a simple configuration). So multiple steps, 1) Find Rules that are similar, and see if they can be merged, 2) abstract the scale of the score, 3) make the changes progressively commiting as we go.

[x] Next to assert_constructed_elements_on_page, can we add a new assert all LegoElements bbounding boxes are the union of their source blocks.

[x] We have multiple places in code that use extract_step_number_value, but they all have this same pattern, get a Candidate or Score object, then look at the blocks, assert the first block is a text, then parse the text. This seems brittle.  If it's a score object, I'd like the Score object to contain the int step number directly. If it's a candidate, I'm not sure, but at a minimum a common helper function, otherwise I'm happy to hear suggestions.

[x] Is src/build_a_long/pdf_extract/tests/domain_invariants_test.py being run, can you verify that, and can we validate if KNOWN_STEP_OVERLAP_FAILURES is still valid

[x]  I also found it odd that src/build_a_long/pdf_extract/classifier/tools/fixtures.json5 is under tools, and not in the fixtures directory with the actual fixtures.  Also should it be named index.json5?

[x] Now that data/BUILD exists, should we move the .gitignore into that directory? and what happens if the sources in the files target don't exist? Say I haven't downloaded the PDFs, will that cause pants to fail?

[ ] We have many constants in the classifiers based on assumptions, for example, how big a Element. Can we parse example files, and use that to generate ranges for the constants.
[ ] Implement a spatial index, (using say shapely) to make various geo-lookups faster.
[ ] I'm considering a change to the Classifier Architecture. We score all blocks, then construct. Construct can fail, because a previous step "stole" a block for its own construction. This greedy approach may be simple to implement, but may not produce a optimal solution. Right now I try and create candidates (based on 1 or more blocks) for each possible item, then process the candidates in sorted order. Perhaps, we take a different appraoch, where candidates are scored on possible source blocks, and we try all permentation of all candidates/blocks, until we find a permentation that globally works.  I ideally want a easy to understand, easy to program, solution, that's robust. Do you have a suggestion how we could change the architecture?
[ ] The Step Number is font size 26, but the hint says smaller for data/75375/6509377.pdf. Can we validate this.
[ ] Many places we use the page dimensions. But we should in many cases substract the progress bar height from the page height. Define a new "workable page area" property, and use that as appropriate.



[x] The Extractor class is quite long. Can we do two things 1) Inside _extract_text_blocks_from_texttrace there is a long section of code to just create a Text object. Can we have a Text.create_from(...) or a helper function create_text_from(...). Whatever is most pythoic. And do this for the other block types. 2) There is this PyMuPDF _convert_drawing_items / clipping code. Should that be moved to a seperate class / file?

[ ] Can we ensure that draw_order is unique on a single page. Then can we update the block.id function to use the draw order, instead of a seperate value.

[x] Page 6509377_page_149 needs fixing
[ ] 6509377_page_015 subassembly has a 2x in the bottom corner, which is actually a image, not Text. We need to implement OCR

---

[x] 6509377_page_016 has a subassembly in the top left corner, that should be a preview
[X] Find the PartImage for each Part
[X] Find the Diagram for each Step
[x] Identify icons. There is a rotate part icon
[x] I would like to add another set of data for the golden tests. Specifically data/40573/6433200.pdf, pages 4,5,7 and 31 
[x] 6433200 page 4 has a new bag element without a bag number. Please classify that correctly.
[x] 6433200 page 31 has step number 43, but it seems to be classified as step 4?
[X] score_details should always store a Score object, that has a single score() method. 
[x]  I would like Candidates to have assertions during constructions, to check some invariants. Specifically, LegoElements that are composite ones (e.g made up of other LegoElements but not of blocks) they should assert source_blocks should be either empty or not. 
[x] Try and extract the xref for images.
[x] Identify the background, and remove
[x] Identify shine on Part (6509377 page 15)
[x] Extract the Progress Bar - progress
[x] Convert all LegoPageElements into the OpenAPI.yaml
[x] We need a arrow classifer
[x] Sometimes the PDF is just a large image per page. I can't process them (yet). Detect that is the case, and skip those kinds of PDFs. We should consider writing out some kind of json, to indicate we attempted (but failed) to support this pdf. 
[x] We have some PDFs that take a very long time to process. I think they have a large amount of blocks, and because we have some N^2 algorithms these are too expensive. Later we may add spatial indexes to reduce the need for the N^2 algorithms - but we can leave that as a TODO. I think an example of this is data/10216/4596701.pdf. Can we investigate why it's slow to process this file e.g `pants run src/build_a_long/pdf_extract:main -- --output-dir debug/all data/10216/4596701.pdf`, and then come up with a way to determine this and skip as apporpriate. 
[x] The include_metadata seems a bit messy now. We should refactor so its not passed around when it mostly defaults to true.
[x] The classifier/classifier_config.py has gotten too long. Can you break it up into per-Classifier configs

[x] Identify substeps - white box around a diagram + optional large 2x + optional step numbers
[x] Identify preview image - top left, white box, final image (scaled down)
[x] I would like to migrate my code base from dataclasses to pydantic. Can you find all dataclasses that seem suitable for migration, and begin to suggest a migration plan. Please do it iteratively, with `pants test :: && pants check ::` after each step
[x] Write a validator that shows the progress bar is always increasing

[x] As a validation rule, I'd like to match all parts, with their part on the catalog page
[x] When I compress my fixtures with bzip, I'd like to ensure the compression is deterministic. Can we set flags, etc, to help ensure the binary identical file is produced every time it's run for the same input.


[x] page 6509377_page_014 has a divider, that helps seperate steps 11-12 from 13-14
[x] Step numbers in sub assemblies are not handled correct yet

[x] 6509377_page_013 is mising a subassembly in the bottom right area of the page


[x] The diagram/step bounds on 6433200_page_031 look a little odd

[x] PartsListClassifier._score does this two pass thing. Scoring everything, then in another loop doing something with those scores. Other classifiers don't look like this, and I think the two loops can be merged into one.

[x] I don't understand _deduplicate_and_assign_diagrams, Can we move the unique constraint to a lot later, e.g during the build. As StepClassifier builds all, it can dedup at that point.  We also might want to re-score Steps / change what diagram goes with each step, as they get claimed.


[x] We recently added BBox.filter_contained, filter_overlapping, expand, and find_best_scoring utilitys.  I'd like to search across the Classifiers for similar methods, and then tweak the Classifiers to use the various helper methods. The goal is to have simplier, easier to read and maintain code.


[x] Some Page(LegoPageElement) properties need comments


[X] Can we add a validator that checks no Element (except Background and ProgressBar) intersects with any Divider

[X] Multiple classifiers have helpful functions such as _score_fill_color. Can we try and dedup their usage, so we can a pool of common helpful functions that can be re-used.

[x] The "Regenerating All Current Fixtures" section in src/build_a_long/pdf_extract/fixtures/README.md seems to have a typo in the command to run. Please fix that.


[x] 6509377_page_017 does not list the trivial section correctly anymore
[x] 6509377_page_045 something is wrong with the new bag number, it seems to have a diagram overlapping it. I think those blocks should either be consumed by the bag number, or by the open bag element.
--- 



Filter full-page backgrounds: Skip Drawing elements whose bbox covers >95% of the page

Improve step count detection: The small white stroke drawings (3-6px) near part counts are probably "x" markers

Investigate duplicate images: Pages 14, 15 have overlapping images at identical positions
Add FlavorText element: For trivia/story text (page 17)
Match more part counts: Some "2x" text blocks aren't being associated with parts
