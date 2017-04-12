
# Functions accessing HATTrie using StringFields without first converting to
# String

function Base.get!(ht::HATTrie, key::StringField, default_::Integer)
    return get!(ht, pointer(key.data, key.part.start), length(key.part),
                default_)
end

