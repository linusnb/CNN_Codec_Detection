?	(
􉼦p@(
􉼦p@!(
􉼦p@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-(
􉼦p@??y??a@1??]??(^@A??ɍ"k??I?ic@*	????C?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??9}?$@!?8???X@)??V`Ȃ$@1?[?ǵX@:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle@?@?ش??!Iy?SG??)?@?ش??1Iy?SG??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchp"???ӧ?!??^?δ??)p"???ӧ?1??^?δ??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism? ????!4y??M??)?=$|?o??1?]T?m???:Preprocessing2F
Iterator::Model?ܙ	?s??!ͩ?c)??)??;???v?1?	#?j???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???H\K@Q	S?i??F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??y??a@??y??a@!??y??a@      ??!       "	??]??(^@??]??(^@!??]??(^@*      ??!       2	??ɍ"k????ɍ"k??!??ɍ"k??:	?ic@?ic@!?ic@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???H\K@y	S?i??F@?"l
@gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??n??0??!??n??0??0"j
?gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropInputConv2DBackpropInputթO?-??!?Җ?8>??0"l
@gradient_tape/sequential/conv_layer1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???P???!T\???#??0";
sequential/conv_layer1/Conv2DConv2D??????!	]?????0"l
@gradient_tape/sequential/conv_layer3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?i??<??!s??8=???0"j
?gradient_tape/sequential/conv_layer2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??_??Ұ?!?ѻ?????0"=
sequential/conv_layer2/Relu_FusedConv2D"?E?S???!\??*????"-
IteratorGetNext/_2_Recv?=D????!9?8+?<??"k
Agradient_tape/sequential/batch_normalization/FusedBatchNormGradV3FusedBatchNormGradV3J??t?v??!⃢)???"\
;gradient_tape/sequential/maxpool_layer1/MaxPool/MaxPoolGradMaxPoolGrad?C?u?@??!Mv?)8??Q      Y@Y????[E@a?tk~X?L@q?IݰaE@y??唝u?"?

both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?42.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 