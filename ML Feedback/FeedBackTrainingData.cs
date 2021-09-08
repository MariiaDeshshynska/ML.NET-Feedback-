using Microsoft.ML.Runtime.Api;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML_Feedback
{
    class FeedBackTrainingData
    {
        [Column(ordinal:"0", name: "Label")]
        public bool IsGood { get; set; }
        [Column(ordinal: "1")]
        public string FeedBackText { get; set; }

        
    }
}
